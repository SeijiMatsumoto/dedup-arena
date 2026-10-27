"""OpenRouter HTTP plumbing shared by every retriever: disk cache, retries, rate limits."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

OPENROUTER = "https://openrouter.ai/api"
RETRY_STATUS = {408, 429, 500, 502, 503, 524, 529}
MAX_ATTEMPTS = 8


class RateLimiter:
    """Spaces request starts so a model never exceeds `rpm` requests per minute."""

    def __init__(self, rpm: float):
        self.interval = 60 / rpm
        self.next_slot = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            delay = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        await asyncio.sleep(delay)


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Honor Retry-After when the server sends it; otherwise exponential backoff, capped."""
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header) + random.random()
        except ValueError:
            pass
    return min(2**attempt, 30) + random.random()


class CacheMiss(RuntimeError):
    """Raised in cache-only mode instead of making a paid call."""


class Client:
    """POST with a disk cache keyed by the exact request, so reruns only pay for what's new.

    Cached hits report the latency measured when the call was first made, flagged `cached`,
    so latency stats can exclude them.
    """

    def __init__(self, cache_dir: Path, http: httpx.AsyncClient, api_key_env: str = "OPENROUTER_API_KEY",
                 cache_only: bool = False):
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        self.http = http
        self.api_key_env = api_key_env
        self.limiters: dict[str, RateLimiter] = {}
        self.gates: dict[str, asyncio.Semaphore] = {}

    def gate(self, name: str, concurrency: int) -> asyncio.Semaphore:
        return self.gates.setdefault(name, asyncio.Semaphore(concurrency))

    async def post(
        self, path: str, body: dict[str, Any], *, limit_key: str | None = None,
        rpm: float | None = None, base_url: str = OPENROUTER,
    ) -> tuple[dict[str, Any], float, bool]:
        url = f"{base_url}{path}"
        key = hashlib.sha256(json.dumps([url, body], sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / key[:2] / f"{key}.json"
        if cache_path.exists():
            hit = json.loads(cache_path.read_text())
            return hit["response"], hit["latency_ms"], True
        if self.cache_only:
            raise CacheMiss(f"not cached: {body.get('model')} {path}")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        limiter = None
        if rpm:
            limiter = self.limiters.setdefault(limit_key or url, RateLimiter(rpm))

        for attempt in range(MAX_ATTEMPTS):
            if limiter:
                await limiter.wait()
            started = perf_counter()
            try:
                response = await self.http.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(retry_delay(None, attempt))
                continue
            latency_ms = (perf_counter() - started) * 1000
            if response.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            payload = response.json()
            if "error" in payload and not any(k in payload for k in ("data", "choices", "results", "answers")):
                # OpenRouter sometimes reports upstream failures inside a 200.
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(retry_delay(None, attempt))
                    continue
                raise RuntimeError(f"Upstream error: {str(payload['error'])[:300]}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"response": payload, "latency_ms": latency_ms}))
            return payload, latency_ms, False
        raise AssertionError("unreachable")


def usage_cost(payload: dict[str, Any], price_in: float | None = None,
               price_out: float | None = None) -> float | None:
    """USD cost OpenRouter reported, or an estimate from per-1M-token prices."""
    usage = payload.get("usage") or {}
    if usage.get("cost") is not None:
        return float(usage["cost"])
    if price_in is None:
        return None
    tokens_in = usage.get("prompt_tokens") or usage.get("total_tokens") or 0
    tokens_out = usage.get("completion_tokens") or 0
    return (tokens_in * price_in + tokens_out * (price_out or 0)) / 1e6

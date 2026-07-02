import os
import time
import logging
import httpx
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# {api_key: (machine_info_dict_or_None, expires_at)}
_cache: dict[str, tuple[dict | None, float]] = {}


async def _fetch_validation(key: str) -> dict | None:
    admin_url = os.environ.get("ADMIN_API_URL", "").rstrip("/")
    secret = os.environ.get("ADMIN_API_SECRET", "")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{admin_url}/api/ml/validate-key",
                params={"key": key},
                headers={"X-ML-Secret": secret},
            )
        data = resp.json()
        return data if data.get("valid") else None
    except Exception as e:
        logger.warning(f"API key validation request failed: {e}")
        return None


async def validate_api_key(key: str) -> dict | None:
    ttl = int(os.environ.get("API_KEY_TTL_SECONDS", "300"))
    now = time.monotonic()

    if key in _cache:
        cached_result, expires_at = _cache[key]
        if now < expires_at:
            return cached_result

    result = await _fetch_validation(key)
    _cache[key] = (result, now + ttl)
    return result


async def require_machine_key(x_api_key: str = Header(...)) -> dict:
    machine = await validate_api_key(x_api_key)
    if not machine:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return machine


def require_admin_key(x_admin_secret: str = Header(...)) -> None:
    expected = os.environ.get("ADMIN_SECRET", "")
    if not expected or x_admin_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid admin secret")

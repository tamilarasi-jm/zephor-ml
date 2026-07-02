import asyncio
import os
import logging
import httpx

logger = logging.getLogger(__name__)

_log_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)


def emit_event(event: dict) -> None:
    try:
        _log_queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.warning("Log queue full — dropping event")


async def _drain_loop() -> None:
    admin_url = os.environ.get("ADMIN_API_URL", "").rstrip("/")
    secret = os.environ.get("ADMIN_API_SECRET", "")

    while True:
        await asyncio.sleep(5)
        if _log_queue.empty() or not admin_url:
            continue

        batch = []
        while not _log_queue.empty():
            try:
                batch.append(_log_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{admin_url}/api/ml/logs",
                    json={"events": batch},
                    headers={"X-ML-Secret": secret},
                )
        except Exception as e:
            logger.warning(f"Log shipping failed (batch of {len(batch)}): {e}")


async def start_log_shipper() -> None:
    asyncio.create_task(_drain_loop())

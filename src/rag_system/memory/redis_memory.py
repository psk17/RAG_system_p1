import logging
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError
from rag_system.memory.models import ChatMessage
from typing import List

logger = logging.getLogger(__name__)

class RedisMemoryStore:
    def __init__(
        self,
        redis: Redis,
        ttl: int = 86400,
    ) -> None:
        self.redis = redis
        self.ttl = ttl
        self._fallback_store: dict[str, List[ChatMessage]] = {}
        self._logged_offline = False

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage,
    ) -> None:
        key = f"chat:{session_id}"
        try:
            await self.redis.rpush(
                key,
                message.model_dump_json(),
            )  # type: ignore[misc]
            await self.redis.expire(
                key,
                self.ttl,
            )  # type: ignore[misc]
        except (ConnectionError, TimeoutError) as e:
            if not self._logged_offline:
                logger.debug(f"Redis offline, falling back to in-memory: {e}")
                self._logged_offline = True
            if session_id not in self._fallback_store:
                self._fallback_store[session_id] = []
            self._fallback_store[session_id].append(message)

    async def get_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> List[ChatMessage]:
        key = f"chat:{session_id}"
        try:
            items = await self.redis.lrange(
                key,
                -limit,
                -1,
            )
            return [
                ChatMessage.model_validate_json(item)
                for item in items
            ]
        except (ConnectionError, TimeoutError) as e:
            if not self._logged_offline:
                logger.debug(f"Redis offline, reading from in-memory: {e}")
                self._logged_offline = True
            history = self._fallback_store.get(session_id, [])
            return history[-limit:]

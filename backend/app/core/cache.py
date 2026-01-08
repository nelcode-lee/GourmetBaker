"""
Caching service for queries and responses
Uses Redis if available, falls back to in-memory cache
"""
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# In-memory cache fallback
_in_memory_cache: Dict[str, Dict[str, Any]] = {}


class CacheService:
    """
    Cache service for storing and retrieving query responses
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache service
        
        Args:
            redis_url: Redis connection URL (optional)
        """
        self.redis_client = None
        self.use_redis = False
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                self.use_redis = True
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory cache: {e}")
                self.use_redis = False
        else:
            logger.info("Using in-memory cache (Redis not configured)")
    
    def _generate_cache_key(
        self,
        query: str,
        document_ids: Optional[list] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None,
        cache_version: str = "v4"  # Version to invalidate old caches (updated for query normalization)
    ) -> str:
        """
        Generate cache key from query and filters
        
        Args:
            query: Query text
            document_ids: Optional document IDs filter
            core_area: Optional core area filter
            factory: Optional factory filter
            cache_version: Cache version to invalidate old entries
            
        Returns:
            Cache key string
        """
        # Create a unique key based on query and filters
        key_parts = [query.lower().strip(), cache_version]  # Include version
        
        if document_ids:
            # Sort document IDs for consistent keys
            sorted_doc_ids = sorted(str(doc_id) for doc_id in document_ids)
            key_parts.append(f"docs:{','.join(sorted_doc_ids)}")
        
        if core_area:
            key_parts.append(f"area:{core_area.lower()}")
        
        if factory:
            key_parts.append(f"factory:{factory.lower()}")
        
        key_string = "|".join(key_parts)
        # Hash for shorter keys
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"query_cache:{key_hash}"
    
    async def get(
        self,
        query: str,
        document_ids: Optional[list] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query
        
        Args:
            query: Query text
            document_ids: Optional document IDs filter
            core_area: Optional core area filter
            factory: Optional factory filter
            
        Returns:
            Cached response dict or None if not found
        """
        cache_key = self._generate_cache_key(query, document_ids, core_area, factory)
        
        try:
            if self.use_redis and self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    logger.info(f"Cache HIT for query: {query[:50]}...")
                    return json.loads(cached_data)
            else:
                # In-memory cache
                if cache_key in _in_memory_cache:
                    cached_entry = _in_memory_cache[cache_key]
                    # Check if expired (simple TTL check)
                    import time
                    if time.time() < cached_entry.get("expires_at", 0):
                        logger.info(f"Cache HIT (in-memory) for query: {query[:50]}...")
                        return cached_entry.get("data")
                    else:
                        # Expired, remove it
                        del _in_memory_cache[cache_key]
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
        
        logger.debug(f"Cache MISS for query: {query[:50]}...")
        return None
    
    async def set(
        self,
        query: str,
        response_data: Dict[str, Any],
        ttl_seconds: int = 3600,  # Default 1 hour
        document_ids: Optional[list] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None
    ):
        """
        Store response in cache
        
        Args:
            query: Query text
            response_data: Response data to cache
            ttl_seconds: Time to live in seconds (default 1 hour)
            document_ids: Optional document IDs filter
            core_area: Optional core area filter
            factory: Optional factory filter
        """
        cache_key = self._generate_cache_key(query, document_ids, core_area, factory)
        
        try:
            if self.use_redis and self.redis_client:
                # Store in Redis with TTL
                json_data = json.dumps(response_data)
                self.redis_client.setex(cache_key, ttl_seconds, json_data)
                logger.info(f"Cached response for query: {query[:50]}... (TTL: {ttl_seconds}s)")
            else:
                # In-memory cache
                import time
                _in_memory_cache[cache_key] = {
                    "data": response_data,
                    "expires_at": time.time() + ttl_seconds
                }
                logger.info(f"Cached response (in-memory) for query: {query[:50]}... (TTL: {ttl_seconds}s)")
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    async def invalidate(
        self,
        query: Optional[str] = None,
        document_ids: Optional[list] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None
    ):
        """
        Invalidate cache entry(s)
        
        Args:
            query: Query text (if None, invalidates all)
            document_ids: Optional document IDs filter
            core_area: Optional core area filter
            factory: Optional factory filter
        """
        if query:
            # Try to invalidate both old and new cache versions
            # Try to invalidate all cache versions
            cache_key_v1 = self._generate_cache_key(query, document_ids, core_area, factory, cache_version="v1")
            cache_key_v2 = self._generate_cache_key(query, document_ids, core_area, factory, cache_version="v2")
            cache_key_v3 = self._generate_cache_key(query, document_ids, core_area, factory, cache_version="v3")
            try:
                if self.use_redis and self.redis_client:
                    self.redis_client.delete(cache_key_v1, cache_key_v2, cache_key_v3)
                else:
                    _in_memory_cache.pop(cache_key_v1, None)
                    _in_memory_cache.pop(cache_key_v2, None)
                    _in_memory_cache.pop(cache_key_v3, None)
                logger.info(f"Invalidated cache for query: {query[:50]}... (both versions)")
            except Exception as e:
                logger.warning(f"Error invalidating cache: {e}")
        else:
            # Invalidate all (use with caution)
            try:
                if self.use_redis and self.redis_client:
                    # Delete all query_cache keys (both old and new versions)
                    keys = self.redis_client.keys("query_cache:*")
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    _in_memory_cache.clear()
                logger.info("Invalidated all cache entries")
            except Exception as e:
                logger.warning(f"Error invalidating all cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            "type": "redis" if self.use_redis else "in-memory",
            "available": self.use_redis or len(_in_memory_cache) > 0
        }
        
        try:
            if self.use_redis and self.redis_client:
                keys = self.redis_client.keys("query_cache:*")
                stats["entries"] = len(keys)
                stats["memory_usage"] = "N/A"  # Would need redis info command
            else:
                stats["entries"] = len(_in_memory_cache)
                stats["memory_usage"] = "N/A"
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            stats["error"] = str(e)
        
        return stats


# Global cache instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """
    Get or create global cache service instance
    
    Returns:
        CacheService instance
    """
    global _cache_service
    if _cache_service is None:
        from app.core.config import get_settings
        settings = get_settings()
        _cache_service = CacheService(redis_url=settings.REDIS_URL)
    return _cache_service


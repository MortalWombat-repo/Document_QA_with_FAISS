import redis
import json
import pickle
import zlib
from typing import Optional, Any
import logging
from pathlib import Path
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # We'll handle serialization ourselves
        )
        self.compress_threshold = 1024 * 1024  # 1MB threshold for compression
        
    def _serialize(self, data: Any) -> bytes:
        """Serialize data with compression for large objects."""
        serialized = pickle.dumps(data)
        if len(serialized) > self.compress_threshold:
            serialized = zlib.compress(serialized)
            return b'z' + serialized  # Prefix to indicate compressed data
        return b'n' + serialized  # Prefix to indicate uncompressed data
    
    def _deserialize(self, serialized: bytes) -> Any:
        """Deserialize data, handling compression if needed."""
        if not serialized:
            return None
        
        prefix = serialized[:1]
        data = serialized[1:]
        
        if prefix == b'z':
            data = zlib.decompress(data)
        return pickle.loads(data)
    
    def _generate_key(self, *parts) -> str:
        """Generate a consistent Redis key from parts."""
        key_parts = [str(part) for part in parts if part is not None]
        return ":".join(key_parts)
    
    def get(self, key_parts: list, default=None) -> Any:
        """Get cached data by key parts."""
        key = self._generate_key(*key_parts)
        try:
            cached = self.redis.get(key)
            if cached is not None:
                return self._deserialize(cached)
            return default
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return default
    
    def set(self, key_parts: list, data: Any, ttl: int = 3600) -> bool:
        """Set data in cache with key parts and optional TTL."""
        key = self._generate_key(*key_parts)
        try:
            serialized = self._serialize(data)
            if ttl > 0:
                return self.redis.setex(key, ttl, serialized)
            return self.redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False
    
    def delete(self, *key_parts) -> bool:
        """Delete cached data by key parts."""
        key = self._generate_key(*key_parts)
        try:
            return self.redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Error deleting cache: {str(e)}")
            return False
    
    def get_file_hash(self, file_path: Path) -> Optional[str]:
        """Get the hash of a file for cache invalidation."""
        try:
            if not file_path.exists():
                return None
            return self._generate_key("filehash", str(file_path), self._calculate_file_hash(file_path))
        except Exception as e:
            logger.error(f"Error getting file hash: {str(e)}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)  # Read in 64k chunks
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

# Global Redis cache instance
redis_cache = RedisCache(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    password=os.getenv("REDIS_PASSWORD")
)
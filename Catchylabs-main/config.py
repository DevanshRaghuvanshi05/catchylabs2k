import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Connection
# Must be provided via DATABASE_URL environment variable (never hard-coded).
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# API Keys — must be provided via environment variables.
# NEVER add default values here; doing so exposes live credentials in source control.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")

# Model Configuration
GEMINI_MODEL_NAME = "gemini-3.5-flash"
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# Upstash Redis Configuration — must be provided via environment variables.
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_URL = os.environ.get("REDIS_URL", "")

# Demo Mode (default off)
DEMO_MODE = os.environ.get("DEMO_MODE", "0") in ("1", "true", "True", "yes", "YES")

# Bring Your Own API Key (BYOK) Fallback Policy
# When false (default), requests without user-configured keys fail with a clear prompt.
# When true, explicitly allows fallback to platform GEMINI_API_KEY / SERPAPI_API_KEY.
ALLOW_PLATFORM_KEY_FALLBACK = os.environ.get("ALLOW_PLATFORM_KEY_FALLBACK", "0").lower() in ("1", "true", "yes")


def get_redis_connection():
    """Returns a connected redis.Redis instance configured for SSL/TLS if rediss:// URL."""
    import redis
    return redis.Redis.from_url(REDIS_URL)

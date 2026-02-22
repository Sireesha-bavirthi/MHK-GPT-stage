from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response

# Initialize limiter with remote address as key
limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiting(app):
    """
    Setup rate limiting for the application.
    Adds the rate limit exception handler.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

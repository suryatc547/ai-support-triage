import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .controllers.ticket_controller import router as ticket_router
from .logging_config import setup_logging
from .middleware import APIKeyMiddleware, TraceIdMiddleware
from .models import database

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise application resources on startup and clean up on shutdown."""
    database.init_db()
    logger.info("Database initialised")
    yield


app = FastAPI(title="Support Ticket API", lifespan=lifespan)

# Note: Starlette nests middleware so the *last* added runs outermost. Adding
# TraceIdMiddleware after APIKeyMiddleware keeps it outermost so even 401s are
# traceable.
app.add_middleware(APIKeyMiddleware)
app.add_middleware(TraceIdMiddleware)

app.include_router(ticket_router)

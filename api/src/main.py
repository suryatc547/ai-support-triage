import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .controllers.ticket_controller import router as ticket_router
from .models import database

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise application resources on startup and clean up on shutdown."""
    database.init_db()
    logger.info("Database initialised")
    yield


app = FastAPI(title="Support Ticket API", lifespan=lifespan)

app.include_router(ticket_router)

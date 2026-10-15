from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from configs.settings import settings
from configs import app_logger as logger, LogErrLevel
from pipeline.api import router as pps_router
from extraction.api import router as extraction_router
from viewer.api import router as viewer_router
from alignment.api import router as alignment_router


# from app.core.middlewares import apply_middlewares

logger.info("Starting PPStructure Pipeline ......")

# Since on event startup & shutdown is deprecated using the lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the startup events
    logger.info("Initializing at Startup")
    # Open the DB async pool on startup and close it on shutdown
    # TBD if required
    yield
    # Clean up the and release the resources before shutdown
    # TBD if required
    logger.info("Cleaning up at Shutdown")
    #TBD


app = FastAPI(lifespan=lifespan, root_path=settings.PPS_PIPELINE_API_ROOT_PATH)
# app = apply_middlewares(app)

# Allow Cross Origin Request. Later allow origins has to be a list of client server ["http://localhost:5000",...]
app.add_middleware(
  CORSMiddleware,
  allow_origins = ["*"],
  allow_methods = ["*"],
  allow_headers = ["*"]
)

# This covers all custom exceptions defined in exception class and all other generic errors.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException ):
    logger.debug(f"Error: Base URL:{request.base_url}")
    logger.debug(f"Error: URL : {request.url}")
    logger.error(str(exc), stack_info=LogErrLevel.STACK, stacklevel=1)
    return JSONResponse(content=str(exc.detail), status_code=exc.status_code)

# This handles the Pydantic validation errors for input
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc ):
    logger.error(str(exc), stack_info=LogErrLevel.STACK, stacklevel=1)
    return JSONResponse(content=f"Bad Request, check the input data. {str(exc)}", status_code=400)

# This handles all other exceptions
@app.exception_handler(Exception)
async def validation_exception_handler(request, exc ):
    logger.error(exc, stack_info=LogErrLevel.STACK, stacklevel=1)
    return JSONResponse(content="Internal Server Error. Please report it to support team.", status_code=500)

app.include_router(pps_router)
app.include_router(extraction_router)
app.include_router(viewer_router)
app.include_router(alignment_router)

logger.info(f"Settings for the applications are \n {settings.model_dump()} \n")
logger.info("PPStructure pipeline started .....")
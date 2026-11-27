import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import get_workflow_service, router as api_v1_router
from app.core.config import get_settings
from app.core.exceptions import VeriSTEMError
from app.core.logging import configure_logging, correlation_id_var, get_logger
from app.core.tracing import configure_tracing
from app.database.session import init_db
from app.schemas.models import ErrorResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings.otel_service_name)
    await init_db()
    logger.info("veristem_startup", version=settings.app_version, model=settings.gemini_model)
    yield
    await get_workflow_service().shutdown()
    logger.info("veristem_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Autonomous, Verifiable STEM Problem-Solving and Scientific Reasoning Platform",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        corr_id = request.headers.get("X-Correlation-ID") or f"corr_{uuid.uuid4().hex[:12]}"
        token = correlation_id_var.set(corr_id)
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = corr_id
            return response
        finally:
            correlation_id_var.reset(token)

    @app.exception_handler(VeriSTEMError)
    async def veristem_exception_handler(_: Request, exc: VeriSTEMError) -> JSONResponse:
        err = ErrorResponse(
            error_code=exc.code,
            message=exc.message,
            correlation_id=correlation_id_var.get(),
        )
        return JSONResponse(status_code=exc.status_code, content=err.model_dump(mode="json"))

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        err = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=str(exc),
            correlation_id=correlation_id_var.get(),
        )
        return JSONResponse(status_code=422, content=err.model_dump(mode="json"))

    app.include_router(api_v1_router)
    return app


app = create_app()

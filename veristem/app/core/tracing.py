from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


def configure_tracing(service_name: str = "veristem-api") -> trace.Tracer:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


@contextmanager
def trace_span(name: str, attributes: dict[str, str | int | float | bool] | None = None) -> Iterator[trace.Span]:
    tracer = trace.get_tracer("veristem")
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span

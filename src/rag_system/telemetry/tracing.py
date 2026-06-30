from typing import Any, Optional

try:
    from opentelemetry import trace
    _tracer = trace.get_tracer("rag_system")
except ImportError:
    class Span:
        def __enter__(self) -> "Span":
            return self

        def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
            pass

        def set_attribute(self, key: str, value: Any) -> None:
            pass

        def is_recording(self) -> bool:
            return True

    class MockTracer:
        def __init__(self, name: str) -> None:
            self.name = name

        def start_as_current_span(self, name: str) -> Span:
            return Span()

    _tracer = MockTracer("rag_system")

tracer: Any = _tracer

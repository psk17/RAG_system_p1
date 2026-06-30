import contextlib
from typing import ContextManager, Generator

try:
    from prometheus_client import Counter as PromCounter, Histogram as PromHistogram
except ImportError:
    class _CounterFallback:
        def __init__(self, name: str, description: str) -> None:
            class ValueHolder:
                def __init__(self) -> None:
                    self.val: int = 0
                def get(self) -> int:
                    return self.val
            self._value = ValueHolder()
        def inc(self, amount: int = 1) -> None:
            self._value.val += amount
    class _HistogramFallback:
        def __init__(self, name: str, description: str) -> None:
            class SumHolder:
                def __init__(self) -> None:
                    self.val: float = 0.0
                def get(self) -> float:
                    return self.val
            self._sum = SumHolder()
        def time(self) -> ContextManager[None]:
            @contextlib.contextmanager
            def timer() -> Generator[None, None, None]:
                yield
            return timer()
    
    PromCounter = _CounterFallback
    PromHistogram = _HistogramFallback

Counter = PromCounter
Histogram = PromHistogram

QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total Queries",
)

QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "Query Duration",
)

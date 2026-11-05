"""Client-side timeout for drivers that block until the query finishes."""

import threading
from collections.abc import Callable
from types import TracebackType


class Deadline:
    """Call cancel from a background thread if the with-block outlives timeout_s.

    Check expired after a failure to tell a timeout apart from other errors. A
    query can also finish just as the timer fires, so only consult expired when
    the query actually failed.
    """

    def __init__(self, timeout_s: float, cancel: Callable[[], None]) -> None:
        self._cancel = cancel
        self._timer = threading.Timer(timeout_s, self._fire)
        # Daemon so a stuck timer never keeps the process alive at exit.
        self._timer.daemon = True
        self.expired = False

    def _fire(self) -> None:
        self.expired = True
        self._cancel()

    def __enter__(self) -> "Deadline":
        self._timer.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._timer.cancel()

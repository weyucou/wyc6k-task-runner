# Copyright (c) 2026 MNKT
"""Tests for marvin.worker — the in-process SQS worker loop."""

import logging
import signal
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from marvin import worker

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_shutdown_state() -> Iterator[None]:
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    yield
    for signum, handler in handlers.items():
        signal.signal(signum, handler)
    worker._shutdown.clear()


class TestMain:
    @pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
    def test_signal_stops_after_current_poll(self, signum) -> None:
        polls: list[object] = []

        def poll(sqs: object) -> None:
            polls.append(sqs)
            signal.raise_signal(signum)

        with (
            patch("marvin.worker._get_sqs_client", return_value="sqs-client"),
            patch("marvin.worker.poll_once", side_effect=poll),
        ):
            worker.main()
        assert polls == ["sqs-client"]

    def test_poll_error_is_logged_and_polling_continues(self, caplog) -> None:
        polls: list[object] = []

        def poll(sqs: object) -> None:
            polls.append(sqs)
            if len(polls) == 1:
                raise RuntimeError("receive failed")
            signal.raise_signal(signal.SIGTERM)

        with (
            patch("marvin.worker._get_sqs_client", return_value="sqs-client"),
            patch("marvin.worker.poll_once", side_effect=poll),
            caplog.at_level(logging.ERROR, logger="marvin.worker"),
        ):
            worker.main()
        assert len(polls) == 2
        [record] = caplog.records
        assert record.getMessage() == "Poll error"
        assert record.exc_info is not None
        assert str(record.exc_info[1]) == "receive failed"

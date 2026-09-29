"""call_with_s3_retry: transient transport failures retry with backoff, everything else raises.

Regression for a checkpoint resume that died on aiohttp.ClientPayloadError ("Not enough data to
satisfy content length header") after 29 minutes of downloading — the wrapper only handled
expired-token ClientErrors.

Run:
    uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/test_s3_retry.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import aiohttp
import pytest

from skyrl.backends.skyrl_train.utils.io import s3fs as s3fs_mod
from skyrl.backends.skyrl_train.utils.io.s3fs import call_with_s3_retry


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(s3fs_mod.time, "sleep", sleeps.append)
    return sleeps


def flaky(failures, error):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise error
        return "ok"

    fn.calls = calls
    return fn


def test_truncated_payload_is_retried(no_sleep):
    error = aiohttp.ClientPayloadError("Response payload is not completed")
    fn = flaky(2, error)
    assert call_with_s3_retry(MagicMock(), fn) == "ok"
    assert fn.calls["n"] == 3
    assert no_sleep == [2.0, 4.0], "exponential backoff"


def test_gives_up_after_max_retries(no_sleep):
    error = aiohttp.ServerDisconnectedError()
    fn = flaky(99, error)
    with pytest.raises(aiohttp.ServerDisconnectedError):
        call_with_s3_retry(MagicMock(), fn)
    assert fn.calls["n"] == 1 + s3fs_mod._TRANSIENT_MAX_RETRIES


def test_non_transient_errors_raise_immediately(no_sleep):
    fn = flaky(99, ValueError("bad key"))
    with pytest.raises(ValueError):
        call_with_s3_retry(MagicMock(), fn)
    assert fn.calls["n"] == 1
    assert no_sleep == []


def test_connection_reset_is_transient(no_sleep):
    fn = flaky(1, ConnectionResetError("peer reset"))
    assert call_with_s3_retry(MagicMock(), fn) == "ok"
    assert fn.calls["n"] == 2


def test_expired_token_path_still_refreshes(no_sleep):
    from skyrl.backends.skyrl_train.utils.io.s3fs import ClientError

    try:
        error = ClientError({"Error": {"Code": "ExpiredToken"}}, "GetObject")
    except TypeError:  # fallback stub type when botocore is absent
        error = ClientError()
        error.response = {"Error": {"Code": "ExpiredToken"}}
    fn = flaky(1, error)
    fs = MagicMock()
    assert call_with_s3_retry(fs, fn) == "ok"
    fs.connect.assert_called_once_with(refresh=True)

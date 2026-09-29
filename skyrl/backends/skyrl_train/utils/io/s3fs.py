import time
from datetime import datetime, timedelta, timezone

import fsspec
from loguru import logger

# Optional AWS deps (present when s3fs is installed)
try:
    import botocore.session as _botocore_session
    from botocore.exceptions import ClientError

    _HAS_BOTOCORE = True
except Exception:
    _HAS_BOTOCORE = False

    class ClientError(Exception):  # fallback type
        pass


_S3_FS = None  # type: ignore


def get_s3_fs():
    """Return a cached S3 filesystem instance, creating it once."""
    global _S3_FS
    if _S3_FS is None:
        _S3_FS = fsspec.filesystem("s3")
    return _S3_FS


def s3_expiry_time():
    """Return botocore credential expiry (datetime in UTC) or None."""
    if not _HAS_BOTOCORE:
        return None
    try:
        sess = _botocore_session.get_session()
        creds = sess.get_credentials()
        if not creds:
            return None
        return getattr(creds, "expiry_time", None) or getattr(creds, "_expiry_time", None)
    except Exception:
        return None


def s3_refresh_if_expiring(fs) -> None:
    """
    Simple refresh:
    - If expiry exists and is within 300s (or past), refresh with fs.connect(refresh=True).
    - Otherwise, do nothing.
    """
    exp = s3_expiry_time()
    if not exp:
        return
    now = datetime.now(timezone.utc)
    if now >= exp - timedelta(seconds=300):
        try:
            fs.connect(refresh=True)  # rebuild session
        except Exception:
            pass


def _transient_transport_errors() -> tuple:
    """Exception types worth retrying: the connection died, not the request being wrong.

    Built lazily because aiohttp is only guaranteed present alongside s3fs. A multi-gigabyte
    checkpoint download holds sockets open for many minutes; under fd pressure or endpoint
    hiccups (GCS's S3-compat interop included) a chunk dies mid-stream as ClientPayloadError
    ("Not enough data to satisfy content length header") — observed killing a resume that had
    already spent 29 minutes downloading.
    """
    errors: list = [ConnectionError, TimeoutError]
    try:
        import aiohttp

        errors += [
            aiohttp.ClientPayloadError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectorError,
            aiohttp.ServerTimeoutError,
        ]
    except ImportError:
        pass
    return tuple(errors)


_TRANSIENT_MAX_RETRIES = 4
_TRANSIENT_BACKOFF_S = 2.0


def call_with_s3_retry(fs, fn, *args, **kwargs):
    """
    Wrapper for calling an S3 method.

    - ExpiredToken and friends: force one credential refresh and retry.
    - Transient transport failures (truncated payload, reset, disconnect, timeout): retry up to
      ``_TRANSIENT_MAX_RETRIES`` times with exponential backoff. Anything else raises immediately.
    """
    transient = _transient_transport_errors()
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except ClientError as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if code in {"ExpiredToken", "ExpiredTokenException", "RequestExpired"} and hasattr(fs, "connect"):
                try:
                    fs.connect(refresh=True)
                except Exception:
                    pass
                return fn(*args, **kwargs)
            raise
        except transient as e:
            attempt += 1
            if attempt > _TRANSIENT_MAX_RETRIES:
                raise
            wait = _TRANSIENT_BACKOFF_S * (2 ** (attempt - 1))
            logger.warning(
                f"transient S3 transport failure ({type(e).__name__}: {str(e)[:200]}); "
                f"retry {attempt}/{_TRANSIENT_MAX_RETRIES} in {wait:.0f}s"
            )
            time.sleep(wait)

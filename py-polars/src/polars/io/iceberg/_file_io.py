"""A PyIceberg FileIO that refreshes vended credentials, for the PyIceberg reader."""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any

from pyiceberg.io.pyarrow import PyArrowFileIO

from polars.io.iceberg._cache import _is_credential_property
from polars.io.iceberg._credentials import provider_credentials_to_iceberg_properties

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyarrow.fs import FileSystem
    from pyiceberg.table import Table
    from pyiceberg.typedef import Properties

    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderIceberg,
    )


class RefreshingPyArrowFileIO(PyArrowFileIO):
    """
    A `PyArrowFileIO` whose credentials come from a credential provider.

    File systems are rebuilt when the provider returns new credentials.
    """

    def __init__(
        self, properties: Properties, provider: CredentialProviderIceberg
    ) -> None:
        super().__init__(dict(properties))
        self._provider = provider
        self._init_refresh_state()

    def _init_refresh_state(self) -> None:
        self._lock = threading.Lock()
        self._fs_cache: dict[tuple[str, str | None], FileSystem] = {}
        self._last_credentials: dict[str, str] | None = None
        # Replaces the parent's `lru_cache`, which never sees new credentials.
        self.fs_by_scheme = self._refreshing_fs_by_scheme

    def _refreshing_fs_by_scheme(
        self, scheme: str, netloc: str | None = None
    ) -> FileSystem:
        credentials, expiry = self._provider()
        # PyIceberg reads files from a thread pool.
        with self._lock:
            if credentials != self._last_credentials:
                for k in provider_credentials_to_iceberg_properties(
                    self._last_credentials or {}, self._provider.scheme
                ):
                    self.properties.pop(k, None)
                self.properties.update(
                    provider_credentials_to_iceberg_properties(
                        credentials, self._provider.scheme, expiry
                    )
                )
                self._fs_cache.clear()
                self._last_credentials = dict(credentials)

            key = (scheme, netloc)
            if (fs := self._fs_cache.get(key)) is None:
                fs = self._fs_cache[key] = self._initialize_fs(scheme, netloc)
            return fs

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        for k in ("_lock", "_fs_cache", "_last_credentials"):
            state.pop(k, None)
        # Credentials are not pickled; the provider installs them again.
        state["properties"] = {
            k: v
            for k, v in state["properties"].items()
            if not _is_credential_property(k)
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._init_refresh_state()


def with_refreshing_io(
    tbl: Table,
    provider: CredentialProviderIceberg,
    static_properties: Mapping[str, str],
) -> Table:
    """
    A shallow copy of `tbl` whose FileIO takes credentials from `provider`.

    The FileIO has the table's FileIO properties without credentials, and
    `static_properties` (Iceberg FileIO properties, e.g. the user's). Tables with
    another FileIO than `PyArrowFileIO` are copied unchanged.
    """
    out = copy.copy(tbl)
    if not isinstance(tbl.io, PyArrowFileIO):
        return out
    properties = {
        k: v for k, v in tbl.io.properties.items() if not _is_credential_property(k)
    }
    properties.update(static_properties)
    out.io = RefreshingPyArrowFileIO(properties, provider)
    return out

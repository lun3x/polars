"""Shared utilities for query monitoring."""

from __future__ import annotations

import os
from typing import Final

MONITORING_ENV_VAR: Final[str] = "POLARS_QUERY_MONITORING"
MONITORING_WORKSPACE_ENV_VAR: Final[str] = "POLARS_QUERY_MONITORING_WORKSPACE"
MONITORING_ORGANIZATION_ENV_VAR: Final[str] = "POLARS_QUERY_MONITORING_ORGANIZATION"

# Minimum version of polars_cloud required to enable local query monitoring
MIN_POLARS_CLOUD_MONITORING_VERSION: Final[str] = "0.13.0"


def monitoring_enabled_globally() -> bool:
    return os.environ.get(MONITORING_ENV_VAR) == "1"


def monitoring_workspace() -> str | None:
    """The Polars Cloud workspace metrics are sent to, `None` for the default one."""
    return os.environ.get(MONITORING_WORKSPACE_ENV_VAR) or None


def monitoring_organization() -> str | None:
    """The Polars Cloud organization metrics are sent to, `None` for the default one."""
    return os.environ.get(MONITORING_ORGANIZATION_ENV_VAR) or None


def activate_monitoring() -> None:
    """Ensure `polars_cloud` is installed and this session is authenticated."""
    from polars._dependencies import import_optional

    import_optional(
        "polars_cloud",
        err_prefix="query monitoring requires the",
        min_version=MIN_POLARS_CLOUD_MONITORING_VERSION,
        min_err_prefix="query monitoring requires",
        install_message=(
            "Please install using the command "
            f"`pip install 'polars-cloud>={MIN_POLARS_CLOUD_MONITORING_VERSION}'`."
        ),
    ).authenticate()

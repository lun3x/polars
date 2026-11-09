"""
Credentials from an Iceberg catalog's FileIO properties.

Maps a table's FileIO properties (including credentials vended by the catalog) to
object store options, and derives the facts a refreshing credential provider needs:
which options are credentials, and when they expire.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from polars._utils.logging import eprint, verbose

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeAlias

    import pyiceberg.catalog

    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderIceberg,
    )
    from polars.io.iceberg._dataset import IcebergCatalogConfig

    import pyiceberg.table

CloudScheme = Literal["aws", "azure", "gcp"]

# A catalog as a plan carries it: a config to build it from, or a name to resolve.
CatalogSource: TypeAlias = "IcebergCatalogConfig | str"

# Expiry of vended credentials, in integer milliseconds since the epoch.
S3_SESSION_TOKEN_EXPIRES_AT_MS = "s3.session-token-expires-at-ms"
# Suffixed with `.<account-host>` by catalogs; `catalog_storage_options` selects the
# location's account.
ADLS_SAS_TOKEN = "adls.sas-token"
ADLS_SAS_EXPIRES_AT_MS = "adls.sas-token-expires-at-ms"
GCS_OAUTH2_TOKEN_EXPIRES_AT = "gcs.oauth2.token-expires-at"

# Credentials are reported as expiring this much early, but never sooner than the
# floor from now, and expire after the default TTL when the catalog vends no expiry.
EXPIRY_MARGIN_SECONDS = 300
EXPIRY_FLOOR_SECONDS = 60
DEFAULT_TTL_SECONDS = 900

_S3_SCHEMES = ("s3", "s3a", "s3n")
_GCS_SCHEMES = ("gs", "gcs")

_AZURE_SCHEMES = ("abfs", "abfss", "adl", "az", "azure", "wasb", "wasbs")

# FileIO properties that PyIceberg honours and that have no object store equivalent,
# by URL scheme.
_UNSUPPORTED_IO_PROPERTIES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("s3", "s3a", "s3n"): (
        # Remote signing (PyIceberg only signs with `s3.signer` set).
        "s3.signer",
        "s3.role-arn",
        "s3.role-session-name",
        "s3.profile-name",
        "client.role-arn",
        "client.role-session-name",
        "client.profile-name",
    ),
    _AZURE_SCHEMES: (
        "adls.connection-string",
        "adls.credential",
        "adls.blob-storage-authority",
        "adls.dfs-storage-authority",
        "adls.blob-storage-scheme",
        "adls.dfs-storage-scheme",
    ),
    ("gs", "gcs"): ("gcs.service.host", "gcs.requester-pays"),
    ("hf",): ("hf.endpoint",),
}

# PyIceberg's fallbacks for S3 properties.
_S3_CLIENT_PROPERTIES = {
    "client.access-key-id": "s3.access-key-id",
    "client.secret-access-key": "s3.secret-access-key",
    "client.session-token": "s3.session-token",
    "client.region": "s3.region",
}

# FileIO property prefixes, by the URL schemes they configure.
_PROPERTY_SCHEMES: dict[str, tuple[str, ...]] = {
    "s3.": ("s3", "s3a", "s3n"),
    "client.": ("s3", "s3a", "s3n"),
    "adls.": _AZURE_SCHEMES,
    "gcs.": ("gs", "gcs"),
    "hf.": ("hf",),
}


def _storage_options(
    properties: Mapping[str, Any],
    location: str,
    user_storage_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Object store options of a table at `location`.

    Its FileIO properties (e.g. vended credentials) with known object store
    equivalents, overridden by the user's storage options. Raises
    `NotImplementedError` as `catalog_storage_options`.
    """
    from polars.io.iceberg._plugin import _is_credential_key

    storage_options = catalog_storage_options(properties, location)
    if user_storage_options is not None:
        user_options = _user_storage_options(user_storage_options)
        # The user's credentials replace the catalog's as a whole: mixing them
        # (e.g. a profile with vended keys) is ambiguous, and rejected by the
        # credential provider.
        if any(_is_credential_key(k) for k in user_options):
            storage_options = {
                k: v
                for k, v in storage_options.items()
                if k not in _CATALOG_CREDENTIAL_KEYS
            }
        # User keys last, in their order: of two aliases, the later one wins.
        storage_options = {
            k: v for k, v in storage_options.items() if k not in user_options
        }
        storage_options.update(user_options)
    return storage_options


def _user_storage_options(user_storage_options: Mapping[str, Any]) -> dict[str, Any]:
    """The user's storage options as object store options."""
    from polars.io.iceberg._dataset import (
        _convert_iceberg_to_object_store_storage_options,
    )

    # Empty values are unset, as for the catalog's.
    return {
        k: v
        for k, v in _convert_iceberg_to_object_store_storage_options(
            dict(user_storage_options)
        ).items()
        if v not in ("", None)
    }


def _native_storage_options(
    table: pyiceberg.table.Table,
    user_storage_options: Mapping[str, Any] | None,
    source: str | None = None,
) -> dict[str, Any] | None:
    """
    Object store options for Polars' own reads and writes of `table`'s files.

    As `_storage_options`, for files at the table's data path, or at `source`
    (a file to read) if that is on another backend. With FileIO properties that
    have no object store equivalent, the user's storage options alone.
    """
    # Where the table writes its data files: stable across scans, unlike a file.
    location = table.properties.get("write.data.path") or table.location()
    # Files on another backend (e.g. added from elsewhere) take its properties.
    if source is not None and _backend(source) != _backend(location):
        location = source
    try:
        return _storage_options(table.io.properties, location, user_storage_options)
    except NotImplementedError as e:
        if verbose():
            eprint(f"{e}; using the user's storage options only")
        if user_storage_options is None:
            return None
        return _user_storage_options(user_storage_options)


def catalog_storage_options(
    properties: Mapping[str, Any], location: str
) -> dict[str, Any]:
    """
    Object store options from a table's FileIO properties.

    Raises `NotImplementedError` for properties that change how storage is accessed
    but have no equivalent, so that PyIceberg plans the scan.
    """
    from polars.io.iceberg._dataset import (
        ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP,
        _convert_iceberg_property_value,
    )

    scheme = _url_scheme(location)
    # Empty values are unset, as in PyIceberg.
    properties = {
        k: v
        for k, v in properties.items()
        if v not in ("", None) and _configures(k, scheme)
    }
    if (host := adls_account_host(location)) is not None:
        properties = _select_adls_account_properties(properties, host)
    for schemes, keys in _UNSUPPORTED_IO_PROPERTIES.items():
        if scheme in schemes and (
            unsupported := sorted(k for k in keys if k in properties)
        ):
            msg = f"iceberg: unsupported: FileIO properties: {unsupported}"
            raise NotImplementedError(msg)
    if scheme in ("s3", "s3a", "s3n"):
        if _property_is_true(properties, "s3.remote-signing-enabled"):
            msg = "iceberg: unsupported: S3 remote signing"
            raise NotImplementedError(msg)
        properties = {
            **{
                s3_key: properties[k]
                for k, s3_key in _S3_CLIENT_PROPERTIES.items()
                if k in properties
            },
            **properties,
        }

    storage_options = {
        (key := ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP[k]): (
            _convert_iceberg_property_value(key, v)
        )
        for k, v in properties.items()
        if k in ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP
    }
    # Anonymous access.
    for schemes, anonymous_key, skip_signature_key in (
        (("s3", "s3a", "s3n"), "s3.anonymous", "aws_skip_signature"),
        (_AZURE_SCHEMES, "adls.anon", "azure_skip_signature"),
    ):
        if scheme in schemes and _property_is_true(properties, anonymous_key):
            storage_options[skip_signature_key] = "true"
    return storage_options


def _backend(location: str) -> tuple[str, ...] | None:
    """The URL schemes of `location`'s backend, as in `_PROPERTY_SCHEMES`."""
    scheme = _url_scheme(location)
    return next((s for s in _PROPERTY_SCHEMES.values() if scheme in s), None)


def _configures(key: str, scheme: str) -> bool:
    """Whether FileIO property `key` applies to `scheme` (other backends' don't)."""
    return all(
        scheme in schemes
        for prefix, schemes in _PROPERTY_SCHEMES.items()
        if key.startswith(prefix)
    )


def _property_is_true(properties: Mapping[str, Any], key: str) -> bool:
    # As PyIceberg's `property_as_bool` (`strtobool`); invalid values are false.
    value = properties.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("y", "yes", "t", "true", "on", "1")


# Object store keys of catalog-provided credentials
# (`catalog_storage_options` values).
_CATALOG_CREDENTIAL_KEYS = frozenset(
    [
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "azure_storage_account_key",
        "azure_storage_sas_key",
        "azure_storage_tenant_id",
        "azure_storage_client_id",
        "azure_storage_client_secret",
        "azure_storage_token",
        "bearer_token",
        "token",
        "aws_skip_signature",
        "azure_skip_signature",
    ]
)


# Credential keys (a subset of `_CATALOG_CREDENTIAL_KEYS`) by cloud that a credential
# provider returns. Other credential keys (e.g. Azure client secrets) are not refreshed.
_PROVIDER_CREDENTIAL_KEYS: dict[CloudScheme, frozenset[str]] = {
    "aws": frozenset(
        ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]
    ),
    "azure": frozenset(
        ["azure_storage_account_key", "azure_storage_sas_key", "azure_storage_token"]
    ),
    "gcp": frozenset(["bearer_token"]),
}

# Object store option keys → credential provider keys, where they differ.
_PROVIDER_KEY_RENAMES = {
    "azure_storage_account_key": "account_key",
    "azure_storage_sas_key": "sas_token",
    "azure_storage_token": "bearer_token",
}


def cloud_scheme(location: str) -> CloudScheme | None:
    """Cloud of a storage location, or `None` if it is not in a supported cloud."""
    scheme = _url_scheme(location)
    if scheme in _S3_SCHEMES:
        return "aws"
    if scheme in _AZURE_SCHEMES:
        return "azure"
    if scheme in _GCS_SCHEMES:
        return "gcp"
    return None


def adls_account_host(location: str) -> str | None:
    """
    Account host of an Azure location.

    E.g. `acct.dfs.core.windows.net` for
    `abfss://container@acct.dfs.core.windows.net/path`; `None` for other locations.
    """
    if _url_scheme(location) not in _AZURE_SCHEMES:
        return None
    authority = location.split("://", 1)[1].split("/", 1)[0]
    if "@" not in authority:
        return None
    return authority.rsplit("@", 1)[1] or None


def split_credential_options(
    options: Mapping[str, Any], scheme: CloudScheme | None
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Split object store options into `(static, credentials)`.

    Credentials are renamed to the keys a credential provider returns. With a
    `scheme`, only that cloud's credential keys are credentials; with `None`, every
    cloud's are.
    """
    credential_keys = (
        _PROVIDER_CREDENTIAL_KEYS[scheme]
        if scheme is not None
        else frozenset().union(*_PROVIDER_CREDENTIAL_KEYS.values())
    )
    static: dict[str, Any] = {}
    credentials: dict[str, str] = {}
    for k, v in options.items():
        if k in credential_keys:
            credentials[_PROVIDER_KEY_RENAMES.get(k, k)] = v
        else:
            static[k] = v
    return static, credentials


# Credential provider keys → Iceberg FileIO properties (the inverse of
# `catalog_storage_options` and `split_credential_options` for credentials).
# `bearer_token` depends on the cloud.
_PROVIDER_KEY_TO_ICEBERG_PROPERTY = {
    "aws_access_key_id": "s3.access-key-id",
    "aws_secret_access_key": "s3.secret-access-key",
    "aws_session_token": "s3.session-token",
    "account_key": "adls.account-key",
    "sas_token": "adls.sas-token",
}


def split_vended_options(
    options: Mapping[str, Any],
    configured_properties: Mapping[str, Any] | None,
    location: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Split object store options into `(static, vended credentials)`.

    As `split_credential_options`, but credentials with the value the catalog's own
    configuration (`configured_properties`) gives are static: PyIceberg merges what
    the catalog vends into the table's FileIO properties, not the catalog's.
    """
    try:
        configured = catalog_storage_options(configured_properties or {}, location)
    except NotImplementedError:
        configured = {}
    configured_keys = {
        k for k, v in options.items() if k in configured and configured[k] == v
    }
    rest, credentials = split_credential_options(
        {k: v for k, v in options.items() if k not in configured_keys},
        cloud_scheme(location),
    )
    static = {k: v for k, v in options.items() if k in configured_keys or k in rest}
    return static, credentials


def provider_credentials_to_iceberg_properties(
    credentials: Mapping[str, str],
    scheme: CloudScheme | None,
    expiry: int | None = None,
) -> dict[str, str]:
    """
    Iceberg FileIO properties for credentials a credential provider returned.

    A bearer token is an ADLS token for `"azure"`, and a GCS token for `"gcp"`, or
    for `None` when no Azure credential is present. A GCS token comes with its
    expiry (PyArrow takes the token only with one): `expiry` (Unix time in seconds,
    as the provider returned it), or `DEFAULT_TTL_SECONDS` from now.
    """
    if scheme == "azure" or (
        scheme is None and ("account_key" in credentials or "sas_token" in credentials)
    ):
        bearer_token_property = "adls.token"
    else:
        bearer_token_property = "gcs.oauth2.token"

    properties: dict[str, str] = {}
    for k, v in credentials.items():
        if k == "bearer_token":
            properties[bearer_token_property] = v
        elif k in _PROVIDER_KEY_TO_ICEBERG_PROPERTY:
            properties[_PROVIDER_KEY_TO_ICEBERG_PROPERTY[k]] = v
    if "gcs.oauth2.token" in properties:
        if expiry is None:
            expiry = int(time.time() + DEFAULT_TTL_SECONDS)
        properties[GCS_OAUTH2_TOKEN_EXPIRES_AT] = str(expiry * 1000)
    return properties


def catalog_credentials(
    *,
    catalog: CatalogSource | None,
    table_identifier: tuple[str, ...] | None,
    location: str,
    catalog_properties: Mapping[str, Any],
    user_storage_options: Mapping[str, Any] | None,
    on_unsupported: Literal["raise", "user_options"] = "raise",
    configured_properties: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], CredentialProviderIceberg | None]:
    """
    Object store options for a table, and a provider of its vended credentials.

    The catalog's options are overridden by the user's; the user's credentials
    replace the catalog's as a whole. Vended credentials are refreshed by a
    `CredentialProviderIceberg` (and left out of the options) when the user gave
    none and the catalog and table are known; otherwise the provider is `None`.

    Credentials with the value the catalog's own configuration
    (`configured_properties`, e.g. `catalog.properties`) gives were not vended, and
    stay in the options.

    Properties without an object store equivalent raise `NotImplementedError`, or
    with `on_unsupported="user_options"`, yield only the user's options.
    """
    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderIceberg,
    )
    from polars.io.iceberg._dataset import (
        _convert_iceberg_to_object_store_storage_options,
    )
    from polars.io.iceberg._plugin import _is_credential_key

    user_options = _convert_iceberg_to_object_store_storage_options(
        dict(user_storage_options or {})
    )

    try:
        options = catalog_storage_options(catalog_properties, location)
    except NotImplementedError as error:
        if on_unsupported != "user_options":
            raise
        if verbose():
            eprint(f"catalog_credentials: {table_identifier}: user options ({error})")
        return user_options, None

    if any(_is_credential_key(k) for k in user_options):
        # Mixing the user's credentials with the catalog's (e.g. a profile with
        # vended keys) is ambiguous, and rejected by the credential provider.
        options = {
            k: v for k, v in options.items() if k not in _CATALOG_CREDENTIAL_KEYS
        }
        options.update(user_options)
        if verbose():
            eprint(f"catalog_credentials: {table_identifier}: user credentials")
        return options, None

    options.update(user_options)
    static, credentials = split_vended_options(options, configured_properties, location)
    if not credentials:
        if verbose():
            eprint(f"catalog_credentials: {table_identifier}: nothing vended")
        return options, None
    if catalog is None or table_identifier is None:
        if verbose():
            eprint(
                f"catalog_credentials: {table_identifier}: vended credentials "
                "without a catalog source, not refreshed"
            )
        return options, None

    if verbose():
        eprint(f"catalog_credentials: {table_identifier}: vended credential provider")
    return static, CredentialProviderIceberg(catalog, table_identifier, location)


def fetch_vended_properties(
    catalog: pyiceberg.catalog.Catalog, identifier: tuple[str, ...], location: str
) -> dict[str, Any]:
    """
    FileIO properties, including vended credentials, for a table at `location`.

    Asks the catalog's `load_credentials` (REST catalogs) if it has one, and falls
    back to loading the table if that fails (e.g. the server lacks the endpoint) or
    returns no credentials (e.g. none for the location's prefix, or the server
    vends them with the table).
    """
    load_credentials = getattr(catalog, "load_credentials", None)
    if callable(load_credentials):
        try:
            properties = dict(load_credentials(identifier, location))
        except Exception as error:
            if verbose():
                eprint(
                    f"fetch_vended_properties: {identifier}: load_credentials "
                    f"failed, falling back to load_table ({error = })"
                )
        else:
            if _has_credentials(properties, location):
                if verbose():
                    eprint(
                        f"fetch_vended_properties: {identifier}: via load_credentials"
                    )
                return properties
            if verbose():
                eprint(
                    f"fetch_vended_properties: {identifier}: load_credentials "
                    "returned no credentials, falling back to load_table"
                )

    if verbose():
        eprint(f"fetch_vended_properties: {identifier}: via load_table")
    return dict(catalog.load_table(identifier).io.properties)


def _has_credentials(properties: Mapping[str, Any], location: str) -> bool:
    """Whether FileIO `properties` hold credentials for `location`."""
    try:
        options = catalog_storage_options(properties, location)
    except NotImplementedError:
        # Not for the fallback to decide; the caller raises.
        return True
    return bool(split_credential_options(options, cloud_scheme(location))[1])


def vended_expiry(
    properties: Mapping[str, Any], location: str, now: float | None = None
) -> int:
    """
    Unix time in seconds at which to refresh credentials vended with `properties`.

    `EXPIRY_MARGIN_SECONDS` before the vended expiry, but no sooner than
    `EXPIRY_FLOOR_SECONDS` from `now`; `DEFAULT_TTL_SECONDS` from `now` without a
    vended expiry.
    """
    if now is None:
        now = time.time()

    expires_at_ms = None
    for key in _expiry_keys(location):
        try:
            expires_at_ms = int(properties[key])
        except (KeyError, TypeError, ValueError):
            continue
        break

    if expires_at_ms is None:
        return int(now + DEFAULT_TTL_SECONDS)
    return int(
        max(
            expires_at_ms // 1000 - EXPIRY_MARGIN_SECONDS,
            now + EXPIRY_FLOOR_SECONDS,
        )
    )


def _expiry_keys(location: str) -> list[str]:
    """Properties holding the credential expiry for `location`, by precedence."""
    scheme = cloud_scheme(location)
    if scheme == "aws":
        return [S3_SESSION_TOKEN_EXPIRES_AT_MS]
    if scheme == "azure":
        host = adls_account_host(location)
        return [
            *([f"{ADLS_SAS_EXPIRES_AT_MS}.{host}"] if host is not None else []),
            ADLS_SAS_EXPIRES_AT_MS,
        ]
    if scheme == "gcp":
        return [GCS_OAUTH2_TOKEN_EXPIRES_AT]
    return []


def _select_adls_account_properties(
    properties: dict[str, Any], host: str
) -> dict[str, Any]:
    """
    `properties` with the SAS token and expiry of account `host` unsuffixed.

    Tokens of other accounts are dropped.
    """
    token_key = f"{ADLS_SAS_TOKEN}.{host}"
    expiry_key = f"{ADLS_SAS_EXPIRES_AT_MS}.{host}"
    selected = {
        k: v
        for k, v in properties.items()
        if not (k.startswith(f"{ADLS_SAS_TOKEN}.") and k != token_key)
    }
    if token_key in selected:
        selected[ADLS_SAS_TOKEN] = selected.pop(token_key)
    if expiry_key in selected:
        selected[ADLS_SAS_EXPIRES_AT_MS] = selected.pop(expiry_key)
    return selected


def _url_scheme(location: str) -> str:
    return location.split("://", 1)[0].lower() if "://" in location else ""

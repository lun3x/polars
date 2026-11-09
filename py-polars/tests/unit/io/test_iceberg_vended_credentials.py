from __future__ import annotations

import pickle
import time
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError

pytest.importorskip("pyiceberg.catalog.sql")

from pyiceberg.catalog.sql import SqlCatalog

from polars.io.cloud._utils import NoPickleOption
from polars.io.cloud.credential_provider import CredentialProviderIceberg
from polars.io.iceberg import _credentials as cred
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    IcebergCatalogTableDescriptor,
    IcebergLoadedTableState,
    IcebergScanTableSerializer,
    IcebergTableWrap,
)

pytestmark = pytest.mark.write_disk

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

S3_LOC = "s3://bucket/wh/ns/t/metadata/00001.metadata.json"
ADLS_LOC = "abfss://c@acct.dfs.core.windows.net/wh/ns/t/metadata/1.metadata.json"
GCS_LOC = "gs://bucket/wh/ns/t/metadata/1.metadata.json"


def test_cloud_scheme() -> None:
    assert cred.cloud_scheme(S3_LOC) == "aws"
    assert cred.cloud_scheme(ADLS_LOC) == "azure"
    assert cred.cloud_scheme(GCS_LOC) == "gcp"
    assert cred.cloud_scheme("file:///tmp/t/metadata/1.metadata.json") is None


def test_adls_account_host() -> None:
    assert cred.adls_account_host(ADLS_LOC) == "acct.dfs.core.windows.net"
    assert cred.adls_account_host(S3_LOC) is None


def test_catalog_storage_options_adls_suffixed_token_matches_account() -> None:
    props = {
        "adls.sas-token.acct.dfs.core.windows.net": "sig=A",
        "adls.sas-token.other.dfs.core.windows.net": "sig=B",
    }
    assert cred.catalog_storage_options(props, ADLS_LOC) == {
        "azure_storage_sas_key": "sig=A"
    }


def test_catalog_storage_options_adls_no_matching_account() -> None:
    props = {"adls.sas-token.other.dfs.core.windows.net": "sig=B"}
    assert cred.catalog_storage_options(props, ADLS_LOC) == {}


def test_split_credential_options_renames_azure_keys() -> None:
    static, creds = cred.split_credential_options(
        {
            "azure_storage_sas_key": "sig",
            "azure_storage_account_name": "acct",
            "azure_skip_signature": "true",
        },
        "azure",
    )
    assert creds == {"sas_token": "sig"}
    assert static == {
        "azure_storage_account_name": "acct",
        "azure_skip_signature": "true",
    }


def test_split_credential_options_aws() -> None:
    static, creds = cred.split_credential_options(
        {
            "aws_access_key_id": "k",
            "aws_secret_access_key": "s",
            "aws_session_token": "t",
            "aws_region": "eu-west-1",
            "aws_endpoint_url": "http://x",
        },
        "aws",
    )
    assert creds == {
        "aws_access_key_id": "k",
        "aws_secret_access_key": "s",
        "aws_session_token": "t",
    }
    assert static == {"aws_region": "eu-west-1", "aws_endpoint_url": "http://x"}


def test_split_credential_options_other_family_is_static() -> None:
    static, creds = cred.split_credential_options(
        {"aws_access_key_id": "k", "bearer_token": "b"}, "gcp"
    )
    assert creds == {"bearer_token": "b"}
    assert static == {"aws_access_key_id": "k"}


def test_split_credential_options_without_scheme() -> None:
    assert cred.split_credential_options(
        {"aws_access_key_id": "k", "aws_secret_access_key": "s", "aws_region": "r"},
        None,
    ) == ({"aws_region": "r"}, {"aws_access_key_id": "k", "aws_secret_access_key": "s"})


@pytest.mark.parametrize(
    ("props", "loc", "expected"),
    [
        (
            {"s3.session-token-expires-at-ms": str((1_000_000 + 3600) * 1000)},
            S3_LOC,
            1_000_000 + 3600 - 300,
        ),
        (
            {"s3.session-token-expires-at-ms": str((1_000_000 + 100) * 1000)},
            S3_LOC,
            1_000_000 + 60,
        ),
        (
            {
                "adls.sas-token-expires-at-ms.acct.dfs.core.windows.net": str(
                    (1_000_000 + 3600) * 1000
                )
            },
            ADLS_LOC,
            1_000_000 + 3600 - 300,
        ),
        (
            {"gcs.oauth2.token-expires-at": str((1_000_000 + 3600) * 1000)},
            GCS_LOC,
            1_000_000 + 3600 - 300,
        ),
        ({}, S3_LOC, 1_000_000 + 900),
    ],
)
def test_vended_expiry(props: dict[str, str], loc: str, expected: int) -> None:
    assert cred.vended_expiry(props, loc, now=1_000_000) == expected


@pytest.fixture(autouse=True)
def _reset_resolver() -> Iterator[None]:
    yield
    pl.Config.set_iceberg_catalog_resolver(None)


@pytest.fixture(autouse=True)
def sqlcatalog_uses_null_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    # the default pool keeps sqlite connections open past the test (ResourceWarning)
    import pyiceberg.catalog.sql as pyiceberg_sql
    from sqlalchemy.pool import NullPool

    create_engine = pyiceberg_sql.create_engine  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pyiceberg_sql,
        "create_engine",
        lambda url, **kwargs: create_engine(url, **{**kwargs, "poolclass": NullPool}),
    )


class VendingCatalog(SqlCatalog):
    """A SqlCatalog whose loaded tables carry vended S3 credentials in io.properties."""

    def __init__(
        self,
        *args: Any,
        expires_in_ms: int | None = 3_600_000,
        with_load_credentials: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vends = 0
        self.expires_in_ms = expires_in_ms
        if with_load_credentials:
            self.load_credentials = self._load_credentials

    def _vended(self) -> dict[str, str]:
        self.vends += 1
        props = {
            "s3.access-key-id": f"AK{self.vends}",
            "s3.secret-access-key": "SK",
            "s3.session-token": f"TOK{self.vends}",
            "s3.region": "eu-west-1",
        }
        if self.expires_in_ms is not None:
            props["s3.session-token-expires-at-ms"] = str(
                int(time.time() * 1000) + self.expires_in_ms
            )
        return props

    def load_table(self, identifier: Any) -> Table:
        tbl = super().load_table(identifier)
        tbl.io.properties.update(self._vended())
        return tbl

    def _load_credentials(self, identifier: Any, location: str) -> dict[str, str]:
        self.credential_calls = getattr(self, "credential_calls", 0) + 1
        self.credential_location = location
        return self._vended()


def new_catalog(
    tmp_path: Path,
    name: str,
    df: pl.DataFrame,
    catalog_class: type[SqlCatalog] = SqlCatalog,
    **kwargs: Any,
) -> SqlCatalog:
    path = tmp_path / name
    path.mkdir()
    catalog = catalog_class(
        name,
        uri=f"sqlite:///{path / 'catalog.db'}",
        warehouse=f"file://{path}",
        **kwargs,
    )
    catalog.create_namespace("ns")
    table = catalog.create_table(("ns", "t"), schema=df.to_arrow().schema)
    df.write_iceberg(table, mode="append")
    return catalog


def new_vending_catalog(tmp_path: Path, **kwargs: Any) -> VendingCatalog:
    catalog = new_catalog(
        tmp_path, "vending", pl.DataFrame({"a": [1, 2]}), VendingCatalog, **kwargs
    )
    assert isinstance(catalog, VendingCatalog)
    # Table setup above loaded the table; count vends from here.
    catalog.vends = 0
    return catalog


@pytest.fixture
def vending(tmp_path: Path) -> VendingCatalog:
    return new_vending_catalog(tmp_path)


@pytest.fixture
def sales(tmp_path: Path) -> SqlCatalog:
    return new_catalog(tmp_path, "sales", pl.DataFrame({"a": [1, 2]}))


def recording(catalog: Catalog) -> tuple[Callable[[str], Catalog], list[str]]:
    names: list[str] = []

    def resolve(name: str) -> Catalog:
        names.append(name)
        return catalog

    return resolve, names


def test_provider_returns_only_credential_keys_and_expiry(
    vending: VendingCatalog,
) -> None:
    p = CredentialProviderIceberg(
        IcebergCatalogConfig.from_catalog(vending), ("ns", "t"), S3_LOC
    )
    creds, expiry = p()
    assert creds == {
        "aws_access_key_id": "AK1",
        "aws_secret_access_key": "SK",
        "aws_session_token": "TOK1",
    }
    assert expiry is not None
    assert time.time() + 3000 < expiry < time.time() + 3400


def test_provider_default_ttl_without_expiry_key(tmp_path: Path) -> None:
    cat = new_vending_catalog(tmp_path, expires_in_ms=None)
    # By name: a config would build a new catalog with the default expiry.
    pl.Config.set_iceberg_catalog_resolver(recording(cat)[0])
    _, expiry = CredentialProviderIceberg("sales", ("ns", "t"), S3_LOC)()
    assert expiry is not None
    assert time.time() + 850 < expiry <= time.time() + 900


def test_provider_prefers_load_credentials(tmp_path: Path) -> None:
    cat = new_vending_catalog(tmp_path, with_load_credentials=True)
    # By name, so that the provider calls this instance.
    pl.Config.set_iceberg_catalog_resolver(recording(cat)[0])
    p = CredentialProviderIceberg("sales", ("ns", "t"), S3_LOC)
    p()
    assert cat.credential_calls == 1
    assert cat.credential_location == S3_LOC
    assert cat.vends == 1


def test_provider_falls_back_to_load_table_when_load_credentials_fails(
    vending: VendingCatalog,
) -> None:
    def fail(identifier: Any, location: str) -> dict[str, str]:
        msg = "no credentials endpoint"
        raise RuntimeError(msg)

    vending.load_credentials = fail  # type: ignore[attr-defined]
    pl.Config.set_iceberg_catalog_resolver(recording(vending)[0])
    creds, _ = CredentialProviderIceberg("sales", ("ns", "t"), S3_LOC)()
    assert creds["aws_access_key_id"] == "AK1"


def test_provider_resolves_name_through_resolver(vending: VendingCatalog) -> None:
    resolver, names = recording(vending)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    creds, _ = CredentialProviderIceberg("sales", ("ns", "t"), S3_LOC)()
    assert names == ["sales"]
    assert creds["aws_access_key_id"] == "AK1"


def test_provider_reuses_catalog_instance_and_refetches_when_cleared(
    vending: VendingCatalog,
) -> None:
    resolver, names = recording(vending)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    p = CredentialProviderIceberg("sales", ("ns", "t"), S3_LOC)
    p()
    p.clear_cached_credentials()
    creds, _ = p()
    assert names == ["sales"]
    assert creds["aws_session_token"] == "TOK2"


def test_provider_pickles_without_credentials(vending: VendingCatalog) -> None:
    p = CredentialProviderIceberg(
        IcebergCatalogConfig.from_catalog(vending), ("ns", "t"), S3_LOC
    )
    p()
    q = pickle.loads(pickle.dumps(p))
    assert q._cached_credentials.get() is None
    assert q._catalog_.get() is None
    assert q.catalog == p.catalog
    assert q.table_identifier == ("ns", "t")
    assert pickle.dumps(p) == pickle.dumps(
        CredentialProviderIceberg(p.catalog, ("ns", "t"), S3_LOC)
    )


def test_provider_raises_when_nothing_vended(sales: SqlCatalog) -> None:
    p = CredentialProviderIceberg(
        IcebergCatalogConfig.from_catalog(sales), ("ns", "t"), S3_LOC
    )
    with pytest.raises(ComputeError, match="vended no storage credentials"):
        p()


def test_provider_accepts_local_location(vending: VendingCatalog) -> None:
    p = CredentialProviderIceberg(
        IcebergCatalogConfig.from_catalog(vending),
        ("ns", "t"),
        "file:///x/1.metadata.json",
    )
    assert p.scheme is None


def test_provider_repr_names_catalog_and_table() -> None:
    r = repr(CredentialProviderIceberg("sales", ("ns", "t"), S3_LOC))
    assert "sales" in r
    assert "ns.t" in r


VENDED = {
    "s3.access-key-id": "AK",
    "s3.secret-access-key": "SK",
    "s3.session-token": "T",
    "s3.region": "eu-west-1",
}
GLUE = {"s3.region": "eu-west-1"}


def test_catalog_credentials_engages_provider() -> None:
    static, provider = cred.catalog_credentials(
        catalog="sales",
        table_identifier=("ns", "t"),
        location=S3_LOC,
        catalog_properties=VENDED,
        user_storage_options=None,
    )
    assert static == {"aws_region": "eu-west-1"}
    assert isinstance(provider, CredentialProviderIceberg)
    assert provider.catalog == "sales"


def test_catalog_credentials_merges_user_non_credential_options() -> None:
    static, provider = cred.catalog_credentials(
        catalog="sales",
        table_identifier=("ns", "t"),
        location=S3_LOC,
        catalog_properties=VENDED,
        user_storage_options={"aws_endpoint_url": "http://x"},
    )
    assert static == {"aws_region": "eu-west-1", "aws_endpoint_url": "http://x"}
    assert provider is not None


def test_catalog_credentials_user_credentials_win() -> None:
    static, provider = cred.catalog_credentials(
        catalog="sales",
        table_identifier=("ns", "t"),
        location=S3_LOC,
        catalog_properties=VENDED,
        user_storage_options={"aws_profile": "me"},
    )
    assert provider is None
    assert "aws_access_key_id" not in static
    assert static["aws_profile"] == "me"


def test_catalog_credentials_nothing_vended() -> None:
    static, provider = cred.catalog_credentials(
        catalog="sales",
        table_identifier=("ns", "t"),
        location=S3_LOC,
        catalog_properties=GLUE,
        user_storage_options=None,
    )
    assert (static, provider) == ({"aws_region": "eu-west-1"}, None)


def test_catalog_credentials_no_source_keeps_static_keys() -> None:
    static, provider = cred.catalog_credentials(
        catalog=None,
        table_identifier=None,
        location=S3_LOC,
        catalog_properties=VENDED,
        user_storage_options=None,
    )
    assert provider is None
    assert static["aws_access_key_id"] == "AK"


def test_catalog_credentials_unsupported_property_policy() -> None:
    props = {**GLUE, "s3.role-arn": "arn:aws:iam::1:role/r"}
    with pytest.raises(NotImplementedError):
        cred.catalog_credentials(
            catalog="sales",
            table_identifier=("ns", "t"),
            location=S3_LOC,
            catalog_properties=props,
            user_storage_options=None,
        )
    static, provider = cred.catalog_credentials(
        catalog="sales",
        table_identifier=("ns", "t"),
        location=S3_LOC,
        catalog_properties=props,
        user_storage_options={"aws_region": "us-east-1"},
        on_unsupported="user_options",
    )
    assert (static, provider) == ({"aws_region": "us-east-1"}, None)


def wrap_for_name(name: str, identifier: str = "ns.t") -> IcebergTableWrap:
    return IcebergTableWrap(
        table_=NoPickleOption(None),
        table_descriptor_=IcebergCatalogTableDescriptor(
            table_identifier=identifier, catalog_config=name
        ),
        serializer=IcebergScanTableSerializer(),
        iceberg_storage_properties=None,
    )


def wrap_for_table(tbl: Table) -> IcebergTableWrap:
    return IcebergTableWrap(
        table_=NoPickleOption(tbl),
        table_descriptor_=None,
        serializer=IcebergScanTableSerializer(),
        iceberg_storage_properties=None,
    )


def roundtrip(wrap: IcebergTableWrap) -> IcebergTableWrap:
    return pickle.loads(pickle.dumps(wrap))  # type: ignore[no-any-return]


def test_loaded_state_from_descriptor_scan(vending: VendingCatalog) -> None:
    resolver, _ = recording(vending)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    wrap = wrap_for_name("sales")
    wrap.get()  # loads the table
    state = roundtrip(wrap).table_descriptor_
    assert isinstance(state, IcebergLoadedTableState)
    assert state.catalog == "sales"
    assert state.table_identifier == ("ns", "t")
    assert state.vended is True
    assert state.metadata_location.endswith(".metadata.json")


def test_loaded_state_from_table_object_carries_catalog_config(
    vending: VendingCatalog,
) -> None:
    tbl = vending.load_table(("ns", "t"))
    state = roundtrip(wrap_for_table(tbl)).table_descriptor_
    assert isinstance(state, IcebergLoadedTableState)
    assert state.catalog == IcebergCatalogConfig.from_catalog(vending)
    assert state.vended is True


def test_loaded_state_static_table_has_no_source(sales: SqlCatalog) -> None:
    from pyiceberg.table import StaticTable

    tbl = StaticTable.from_metadata(sales.load_table(("ns", "t")).metadata_location)
    state = roundtrip(wrap_for_table(tbl)).table_descriptor_
    assert isinstance(state, IcebergLoadedTableState)
    assert state.catalog is None
    assert state.vended is False


def test_loaded_state_glue_like_is_not_vended(sales: SqlCatalog) -> None:
    state = roundtrip(wrap_for_table(sales.load_table(("ns", "t")))).table_descriptor_
    assert isinstance(state, IcebergLoadedTableState)
    assert state.vended is False
    assert state.catalog == IcebergCatalogConfig.from_catalog(sales)


def test_get_on_vended_state_fetches_then_loads_from_metadata_location(
    vending: VendingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver, names = recording(vending)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    loaded = wrap_for_name("sales")
    loaded.get()
    wrap = roundtrip(loaded)
    loads: list[Any] = []

    def load_table(ident: Any) -> Table:
        loads.append(ident)
        return VendingCatalog.load_table(vending, ident)

    monkeypatch.setattr(vending, "load_table", load_table)
    names.clear()
    tbl = wrap.get()
    state = wrap.table_descriptor_
    assert isinstance(state, IcebergLoadedTableState)
    # The resolver is called for credentials, and the table loaded only for them
    # (this catalog has no `load_credentials`).
    assert names == ["sales"]
    assert loads == [("ns", "t")]
    # The pinned snapshot.
    assert tbl.metadata_location == state.metadata_location
    assert tbl.io.properties["s3.access-key-id"].startswith("AK")


def test_get_on_vended_state_surfaces_refusing_resolver(
    vending: VendingCatalog,
) -> None:
    resolver, _ = recording(vending)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    loaded = wrap_for_name("sales")
    loaded.get()
    wrap = roundtrip(loaded)

    def refuse(name: str) -> Catalog:
        msg = f'catalog "{name}" refused'
        raise ComputeError(msg)

    pl.Config.set_iceberg_catalog_resolver(refuse)
    with pytest.raises(ComputeError, match='catalog "sales" refused'):
        wrap.get()


def test_old_str_state_still_loads(sales: SqlCatalog) -> None:
    tbl = sales.load_table(("ns", "t"))
    wrap = IcebergTableWrap(
        table_=NoPickleOption(None),
        table_descriptor_=tbl.metadata_location,
        serializer=IcebergScanTableSerializer(),
        iceberg_storage_properties=None,
    )
    assert wrap.get().metadata_location == tbl.metadata_location


def test_loaded_state_survives_a_second_roundtrip(vending: VendingCatalog) -> None:
    pl.Config.set_iceberg_catalog_resolver(recording(vending)[0])
    loaded = wrap_for_name("sales")
    loaded.get()
    wrap = roundtrip(loaded)
    wrap.get()
    state = roundtrip(wrap).loaded_state()
    assert state == wrap.loaded_state()
    assert state is not None
    assert state.catalog == "sales"


def test_catalog_source() -> None:
    assert wrap_for_name("sales").catalog_source() == ("sales", ("ns", "t"))


@pytest.mark.parametrize(
    ("creds", "scheme", "expected"),
    [
        (
            {"aws_access_key_id": "k", "aws_secret_access_key": "s"},
            "aws",
            {"s3.access-key-id": "k", "s3.secret-access-key": "s"},
        ),
        ({"aws_session_token": "t"}, None, {"s3.session-token": "t"}),
        ({"sas_token": "sig"}, "azure", {"adls.sas-token": "sig"}),
        ({"account_key": "a"}, "azure", {"adls.account-key": "a"}),
        ({"bearer_token": "b"}, "azure", {"adls.token": "b"}),
        ({"bearer_token": "b"}, "gcp", {"gcs.oauth2.token": "b"}),
        ({"bearer_token": "b"}, None, {"gcs.oauth2.token": "b"}),
        (
            {"bearer_token": "b", "sas_token": "sig"},
            None,
            {"adls.token": "b", "adls.sas-token": "sig"},
        ),
    ],
)
def test_provider_credentials_to_iceberg_properties(
    creds: dict[str, str], scheme: cred.CloudScheme | None, expected: dict[str, str]
) -> None:
    assert cred.provider_credentials_to_iceberg_properties(creds, scheme) == expected

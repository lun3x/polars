from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg.catalog.sql")

from polars.io.iceberg import _credentials as cred

pytestmark = pytest.mark.write_disk

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

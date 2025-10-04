from __future__ import annotations
import os
from typing import Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        import streamlit as st
        val = st.secrets.get(key)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)

def is_enabled() -> bool:
    return boto3 is not None and bool(_cfg("S3_BUCKET"))

def _client():
    if not is_enabled():
        return None
    region = _cfg("AWS_REGION")
    if region:
        return boto3.client("s3", region_name=region)  # type: ignore
    return boto3.client("s3")  # type: ignore

def _full_key(key: str) -> str:
    prefix = _cfg("S3_PREFIX", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{key}" if prefix else key

def put_file(local_path: str, key: str) -> bool:
    """
    Upload local_path -> s3://bucket/prefix/key
    """
    cli = _client()
    if not cli:
        return False
    bucket = _cfg("S3_BUCKET")
    try:
        cli.upload_file(local_path, bucket, _full_key(key))
        return True
    except (BotoCoreError, ClientError):
        return False

def get_file(key: str, local_path: str) -> bool:
    cli = _client()
    if not cli:
        return False
    bucket = _cfg("S3_BUCKET")
    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        cli.download_file(bucket, _full_key(key), local_path)
        return True
    except (BotoCoreError, ClientError):
        return False

def exists(key: str) -> bool:
    cli = _client()
    if not cli:
        return False
    bucket = _cfg("S3_BUCKET")
    try:
        cli.head_object(Bucket=bucket, Key=_full_key(key))
        return True
    except (BotoCoreError, ClientError):
        return False

def url(key: str) -> Optional[str]:
    """
    Return a presigned URL (default 1 hour) if enabled.
    """
    cli = _client()
    if not cli:
        return None
    bucket = _cfg("S3_BUCKET")
    try:
        return cli.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": _full_key(key)},
            ExpiresIn=3600,
        )
    except (BotoCoreError, ClientError):
        return None

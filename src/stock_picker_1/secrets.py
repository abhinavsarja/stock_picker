"""Load runtime secrets from AWS Secrets Manager into ``os.environ``.

Design:
- In AWS (AgentCore Runtime, ECS, Lambda, EC2), secrets live in Secrets Manager
  as a single JSON document, e.g. secret name ``stock-picker-1/secrets`` with:
      {
          "SERPER_API_KEY": "...",
          "PUSHOVER_USER": "...",
          "PUSHOVER_TOKEN": "..."
      }
- The AgentCore execution role needs ``secretsmanager:GetSecretValue`` on that
  secret's ARN.
- Locally, you typically set the same variables via ``.env`` / shell exports.
  This loader skips silently if:
    * ``AWS_SECRETS_ID`` env var is not set (local dev), or
    * boto3 / AWS credentials are unavailable, or
    * the secret can't be fetched (logged, not raised).
- Already-set env vars are NEVER overwritten, so a local ``.env`` takes
  precedence over the remote secret during dev.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_logger = logging.getLogger(__name__)

_LOADED = False


def load_secrets_from_aws(secret_id: str | None = None, region: str | None = None) -> dict[str, str]:
    """Fetch a JSON secret from AWS Secrets Manager and populate ``os.environ``.

    Args:
        secret_id: Secret name or ARN. Defaults to ``$AWS_SECRETS_ID``.
        region: AWS region. Defaults to ``$AWS_REGION`` / boto3 session default.

    Returns:
        Dict of the keys that were injected (empty if loading was skipped).
    """
    global _LOADED
    if _LOADED:
        return {}

    secret_id = secret_id or os.environ.get("AWS_SECRETS_ID")
    if not secret_id:
        _logger.debug("AWS_SECRETS_ID not set; skipping Secrets Manager load.")
        _LOADED = True
        return {}

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        _logger.warning("boto3 not installed; skipping Secrets Manager load.")
        _LOADED = True
        return {}

    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("secretsmanager", region_name=region) if region else boto3.client("secretsmanager")

    try:
        resp = client.get_secret_value(SecretId=secret_id)
    except (BotoCoreError, ClientError) as exc:
        _logger.error("Failed to load secret %s from Secrets Manager: %s", secret_id, exc)
        _LOADED = True
        return {}

    raw = resp.get("SecretString")
    if raw is None:
        _logger.warning("Secret %s has no SecretString (binary secrets are not supported).", secret_id)
        _LOADED = True
        return {}

    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        _logger.error("Secret %s is not valid JSON; expected an object of string->string.", secret_id)
        _LOADED = True
        return {}

    if not isinstance(payload, dict):
        _logger.error("Secret %s must be a JSON object, got %s.", secret_id, type(payload).__name__)
        _LOADED = True
        return {}

    injected: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key in os.environ:
            continue
        os.environ[key] = value
        injected[key] = "***"

    _logger.info("Loaded %d secret(s) from %s into environment.", len(injected), secret_id)
    _LOADED = True
    return injected

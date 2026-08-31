from __future__ import annotations

import json
import os
import ssl
import threading
import time
import uuid
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

from skillscope.common.config import AppConfig, LLMConfig
from skillscope.common.io import ensure_directory


class StructuredLLMClient(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> dict[str, Any]:
        """Return a JSON object that matches the caller's expected schema."""


class DisabledLLMClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> dict[str, Any]:
        raise RuntimeError(
            "No LLM client is configured. Set SKILLSCOPE_LLM_PROVIDER, SKILLSCOPE_LLM_API_KEY, "
            "SKILLSCOPE_LLM_BASE_URL, and SKILLSCOPE_LLM_MODEL in the environment or project .env."
        )


class LLMDebugLogger:
    def __init__(self, path: Path, *, preview_chars: int = 500) -> None:
        self.path = path
        self.preview_chars = preview_chars
        self._lock = threading.Lock()

    def log(self, payload: dict[str, Any]) -> None:
        ensure_directory(self.path.parent)
        normalized = dict(payload)
        normalized.setdefault("logged_at", _utc_now())
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    def preview(self, text: str) -> str:
        return text[: self.preview_chars]


class OpenAICompatibleLLMClient:
    MAX_TRANSPORT_ATTEMPTS = 3

    def __init__(self, config: LLMConfig, *, debug_logger: LLMDebugLogger | None = None) -> None:
        self.config = config
        self.debug_logger = debug_logger

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> dict[str, Any]:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        call_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            # Chat Completions deprecated ``max_tokens`` in favor of
            # ``max_completion_tokens``.  GPT-5.1 supports the latter and the
            # limit includes both visible output and any reasoning tokens.
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "X-SkillScope-Schema": schema_name,
        }
        self._log_start(
            call_id=call_id,
            schema_name=schema_name,
            endpoint=endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        started_at = time.monotonic()
        try:
            response_payload = self._request_with_transport_retry(
                endpoint=endpoint,
                payload=payload,
                headers=headers,
            )
            content = self._extract_content(response_payload)
            parsed = self._parse_response_content(content)
        except Exception as exc:  # pragma: no cover - exercised through runtime integration
            runtime_error = self._normalize_error(exc)
            self._log_finish(
                call_id=call_id,
                schema_name=schema_name,
                status="error",
                duration_ms=(time.monotonic() - started_at) * 1000.0,
                error=str(runtime_error),
                response_preview=content if "content" in locals() else None,
            )
            raise runtime_error from exc

        self._log_finish(
            call_id=call_id,
            schema_name=schema_name,
            status="success",
            duration_ms=(time.monotonic() - started_at) * 1000.0,
            response_preview=content,
            parsed_keys=sorted(parsed.keys()),
        )
        return parsed

    def _request_with_transport_retry(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        encoded_payload = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.MAX_TRANSPORT_ATTEMPTS + 1):
            http_request = request.Request(
                endpoint,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(
                    http_request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("LLM endpoint response was not a JSON object.")
                return parsed
            except Exception as exc:
                if (
                    attempt >= self.MAX_TRANSPORT_ATTEMPTS
                    or not self._is_retryable_transport_error(exc)
                ):
                    raise
                time.sleep(0.25 * (2 ** (attempt - 1)))
        raise RuntimeError("LLM transport retry loop ended unexpectedly.")

    @staticmethod
    def _is_retryable_transport_error(exc: Exception) -> bool:
        if isinstance(exc, HTTPError):
            return False
        return isinstance(
            exc,
            (
                RemoteDisconnected,
                URLError,
                TimeoutError,
                ConnectionError,
                ssl.SSLError,
            ),
        )

    def _extract_content(self, response_payload: dict[str, Any]) -> str:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response did not contain any choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_fragments: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_fragments.append(text_value)
            if text_fragments:
                return "\n".join(text_fragments)
        raise RuntimeError("LLM response content was not a string.")

    def _extract_json_object(self, content: str) -> str:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("LLM response did not contain a JSON object.")
        return content[start : end + 1]

    def _parse_response_content(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = json.loads(self._extract_json_object(content))
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response JSON was not an object.")
        return parsed

    def _normalize_error(self, exc: Exception) -> RuntimeError:
        if isinstance(exc, RuntimeError):
            return exc
        if isinstance(exc, HTTPError):
            detail = exc.read().decode("utf-8", errors="replace")
            return RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}")
        if isinstance(exc, URLError):
            return RuntimeError(f"LLM request failed: {exc}")
        if isinstance(exc, json.JSONDecodeError):
            return RuntimeError(f"LLM response JSON parsing failed: {exc}")
        return RuntimeError(f"LLM request failed: {exc}")

    def _log_start(
        self,
        *,
        call_id: str,
        schema_name: str,
        endpoint: str,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        if self.debug_logger is None:
            return
        self.debug_logger.log(
            {
                "phase": "start",
                "call_id": call_id,
                "pid": os.getpid(),
                "provider": self.config.provider,
                "model": self.config.model,
                "schema_name": schema_name,
                "endpoint": endpoint,
                "timeout_seconds": self.config.timeout_seconds,
                "json_mode": self.config.json_mode,
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "system_prompt_preview": self.debug_logger.preview(system_prompt),
                "user_prompt_preview": self.debug_logger.preview(user_prompt),
            }
        )

    def _log_finish(
        self,
        *,
        call_id: str,
        schema_name: str,
        status: str,
        duration_ms: float,
        error: str | None = None,
        response_preview: str | None = None,
        parsed_keys: list[str] | None = None,
    ) -> None:
        if self.debug_logger is None:
            return
        payload: dict[str, Any] = {
            "phase": "finish",
            "call_id": call_id,
            "pid": os.getpid(),
            "provider": self.config.provider,
            "model": self.config.model,
            "schema_name": schema_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
        }
        if error is not None:
            payload["error"] = error
        if response_preview is not None:
            payload["response_preview"] = self.debug_logger.preview(response_preview)
        if parsed_keys is not None:
            payload["parsed_keys"] = parsed_keys
        self.debug_logger.log(payload)


def build_llm_client(config: AppConfig) -> StructuredLLMClient:
    if not config.llm.enabled:
        return DisabledLLMClient()
    provider = config.llm.provider.lower()
    if provider in {"openai", "openai_compatible"}:
        debug_logger = None
        if config.llm.debug_enabled and config.llm.debug_log_path is not None:
            debug_logger = LLMDebugLogger(
                config.llm.debug_log_path,
                preview_chars=config.llm.debug_preview_chars,
            )
        return OpenAICompatibleLLMClient(config.llm, debug_logger=debug_logger)
    raise RuntimeError(f"Unsupported SKILLSCOPE_LLM_PROVIDER: {config.llm.provider}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

from __future__ import annotations

import base64
import binascii
import json
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Protocol

from skillscope.common.models import ResourceFixture
from skillscope.common.sandbox.policy import resolve_within


class MCPToolRunner(Protocol):
    """Minimal adapter for a real MCP client supplied by a deployment."""

    def invoke(
        self,
        *,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FixtureMaterialization:
    fixture_id: str
    server: str
    strategy: str
    artifacts: tuple[str, ...] = ()


class MCPBackedFixtureToolchain:
    """Materialize validated fixture specs through MCP-compatible adapters.

    A production deployment may inject a real ``MCPToolRunner``.  The default
    adapters implement the same bounded operations locally inside the sandbox,
    which keeps tests and offline analysis executable while preserving the
    framework's fixture categories and tool boundaries.
    """

    SERVER_BY_TYPE = {
        "file": "filesystem",
        "text_file": "filesystem",
        "generated_file": "filesystem",
        "existing_file": "filesystem",
        "config": "filesystem",
        "binary_file": "filesystem",
        "git": "filesystem+git+terminal",
        "document": "filesystem+pandoc",
        "image": "placeholder-image",
        "api": "mockloop",
        "env": "sandbox-environment",
    }

    def __init__(self, runner: MCPToolRunner | None = None) -> None:
        self.runner = runner

    @classmethod
    def metadata_for(cls, fixture_type: str, target: str) -> dict[str, Any]:
        normalized = fixture_type.strip().lower()
        metadata: dict[str, Any] = {
            "fixture_backend": "mcp_compatible_adapter",
            "mcp_server": cls.SERVER_BY_TYPE.get(normalized, "filesystem"),
        }
        suffix = Path(target).suffix.lower()
        if normalized == "document" and suffix in {".docx", ".pdf"}:
            metadata["conversion_target"] = suffix.removeprefix(".")
        if normalized == "api":
            metadata.update(
                {
                    "status_code": 200,
                    "mock_api_kind": "openapi_exact_endpoint",
                }
            )
        if normalized == "git":
            metadata["repository_commits"] = 2
        if normalized == "image":
            metadata.update({"image_format": "png", "width": 1, "height": 1})
        return metadata

    def materialize(
        self,
        *,
        sandbox_root: Path,
        fixture: ResourceFixture,
    ) -> FixtureMaterialization:
        sandbox_root = sandbox_root.resolve()
        fixture_type = fixture.fixture_type.strip().lower()
        server = self.SERVER_BY_TYPE.get(fixture_type, "filesystem")
        if self.runner is not None:
            result = self.runner.invoke(
                server=server,
                tool=self._tool_for(fixture_type, fixture.target),
                arguments={
                    "fixture_id": fixture.fixture_id,
                    "target": fixture.target,
                    "content": fixture.content,
                    "metadata": dict(fixture.metadata),
                    "sandbox_root": str(sandbox_root),
                },
            )
            artifacts = result.get("artifacts", []) if isinstance(result, dict) else []
            if not isinstance(artifacts, list) or any(
                not isinstance(item, str) for item in artifacts
            ):
                raise RuntimeError("MCP fixture tool returned an invalid artifact list.")
            return FixtureMaterialization(
                fixture_id=fixture.fixture_id,
                server=server,
                strategy="external_mcp_runner",
                artifacts=tuple(artifacts),
            )

        artifacts = self._materialize_locally(
            sandbox_root=sandbox_root,
            fixture=fixture,
        )
        return FixtureMaterialization(
            fixture_id=fixture.fixture_id,
            server=server,
            strategy="sandboxed_mcp_compatible_adapter",
            artifacts=tuple(artifacts),
        )

    def _materialize_locally(
        self,
        *,
        sandbox_root: Path,
        fixture: ResourceFixture,
    ) -> list[str]:
        fixture_type = fixture.fixture_type.strip().lower()
        if fixture_type in {"api", "env"}:
            # Exact API responses and environment values are served from the
            # fixture manifest by the instrumented runtime.
            return []
        target = resolve_within(sandbox_root, fixture.target)
        if fixture_type == "git":
            return self._materialize_git(target, fixture)
        if fixture_type == "document":
            return self._materialize_document(target, fixture.content or "")
        if fixture_type == "image":
            return self._materialize_image(target, fixture)

        target.parent.mkdir(parents=True, exist_ok=True)
        if fixture.metadata.get("encoding") == "base64" or fixture_type == "binary_file":
            try:
                payload = base64.b64decode(fixture.content or "", validate=True)
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError(
                    f"Fixture {fixture.fixture_id} contains invalid base64 data."
                ) from exc
            target.write_bytes(payload)
        else:
            target.write_text(fixture.content or "", encoding="utf-8")
        return [str(target.relative_to(sandbox_root))]

    def _materialize_git(
        self,
        repository_root: Path,
        fixture: ResourceFixture,
    ) -> list[str]:
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("A git fixture was requested, but git is unavailable.")
        repository_root.mkdir(parents=True, exist_ok=True)

        def run(*arguments: str) -> None:
            subprocess.run(
                [git, *arguments],
                cwd=repository_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        run("init", "--quiet")
        run("config", "user.name", "SkillScope Fixture")
        run("config", "user.email", "fixture@skillscope.invalid")
        readme = repository_root / "README.md"
        readme.write_text(
            fixture.content or "# Synthetic repository\n\nInitial fixture.\n",
            encoding="utf-8",
        )
        run("add", "README.md")
        run("commit", "--quiet", "-m", "Initial fixture")
        sample = repository_root / "sample.txt"
        sample.write_text("Second deterministic fixture revision.\n", encoding="utf-8")
        run("add", "sample.txt")
        run("commit", "--quiet", "-m", "Add sample input")
        return [str(repository_root), str(readme), str(sample)]

    def _materialize_document(self, target: Path, text: str) -> list[str]:
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".docx":
            self._write_docx(target, text)
        elif suffix == ".pdf":
            target.write_bytes(self._minimal_pdf(text))
        else:
            target.write_text(text, encoding="utf-8")
        return [str(target)]

    def _materialize_image(
        self,
        target: Path,
        fixture: ResourceFixture,
    ) -> list[str]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if fixture.content:
            try:
                payload = base64.b64decode(fixture.content, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError(
                    f"Fixture {fixture.fixture_id} contains invalid image data."
                ) from exc
        else:
            payload = self._placeholder_png()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Image fixture must contain a valid PNG signature.")
        target.write_bytes(payload)
        return [str(target)]

    def _tool_for(self, fixture_type: str, target: str) -> str:
        if fixture_type == "git":
            return "create_repository_with_commits"
        if fixture_type == "document" and Path(target).suffix.lower() in {".docx", ".pdf"}:
            return "convert_document"
        if fixture_type == "image":
            return "create_placeholder_png"
        if fixture_type == "api":
            return "create_openapi_mock"
        if fixture_type == "env":
            return "bind_environment_value"
        return "write_file"

    def _write_docx(self, target: Path, text: str) -> None:
        import zipfile

        paragraphs = "".join(
            f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>"
            for line in (text.splitlines() or [""])
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        relationships = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>"
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{paragraphs}<w:sectPr/></w:body></w:document>"
        )
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", relationships)
            archive.writestr("word/document.xml", document)

    def _minimal_pdf(self, text: str) -> bytes:
        safe = " ".join(text.split())[:500].replace("\\", "\\\\")
        safe = safe.replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 10 Tf 36 756 Td ({safe}) Tj ET".encode("latin-1", "replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        output = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{index} 0 obj\n".encode("ascii"))
            output.extend(obj + b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
        )
        return bytes(output)

    def _placeholder_png(self) -> bytes:
        def chunk(name: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + name
                + data
                + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
            )

        raw = b"\x00\x3a\x7b\xc4\xff"
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


def fixture_manifest_payload(fixture: ResourceFixture) -> dict[str, Any]:
    payload = {
        "fixture_id": fixture.fixture_id,
        "fixture_type": fixture.fixture_type,
        "target": fixture.target,
        "content": fixture.content,
        "source": fixture.source,
        "required": fixture.required,
        "metadata": {
            **MCPBackedFixtureToolchain.metadata_for(
                fixture.fixture_type,
                fixture.target,
            ),
            **fixture.metadata,
        },
    }
    if fixture.fixture_type == "api" and not payload["content"]:
        payload["content"] = json.dumps({"fixture": True, "status": "ok"})
    return payload

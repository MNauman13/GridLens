"""Content-checked artifact bundles with safe relative file references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from gridlens.contracts import StrictRecord


class ArtifactFile(StrictRecord):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ArtifactManifest(StrictRecord):
    schema_version: Literal["gridlens_artifact_v1"] = "gridlens_artifact_v1"
    kind: str = Field(min_length=1)
    files: tuple[ArtifactFile, ...] = Field(min_length=1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(root: Path, relative: str) -> Path:
    base = root.resolve()
    member = (base / relative).resolve()
    # Reject Windows path syntax even when verifying on Linux.
    if Path(relative).is_absolute() or ":" in relative or "\\" in relative or not member.is_relative_to(base) or member == base:
        raise ValueError("Artifact path must stay inside its bundle")
    return member


def build_manifest(root: Path, files: list[str], *, kind: str) -> ArtifactManifest:
    if len(set(files)) != len(files):
        raise ValueError("Duplicate artifact members")
    records = []
    for relative in sorted(files):
        member = safe_member(root, relative)
        records.append(ArtifactFile(path=relative, sha256=sha256(member), size_bytes=member.stat().st_size))
    return ArtifactManifest(kind=kind, files=tuple(records))


def verify_manifest(root: Path, manifest: ArtifactManifest) -> None:
    paths = [record.path for record in manifest.files]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact members")
    for record in manifest.files:
        member = safe_member(root, record.path)
        if not member.is_file() or member.stat().st_size != record.size_bytes or sha256(member) != record.sha256:
            raise ValueError(f"Corrupt or missing artifact: {record.path}")


def manifest_id(manifest: ArtifactManifest) -> str:
    canonical = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

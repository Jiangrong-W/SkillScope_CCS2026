from __future__ import annotations

import sysconfig
from pathlib import Path


class PromptAssetLoader:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def load(self, asset_path: str) -> str:
        source_path = self.project_root / asset_path
        if source_path.is_file():
            return source_path.read_text(encoding="utf-8")

        installed_path = Path(sysconfig.get_path("data")) / "share" / "skillscope" / asset_path
        if installed_path.is_file():
            return installed_path.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"Prompt asset {asset_path!r} was not found in {self.project_root} "
            f"or the installed SkillScope data directory."
        )

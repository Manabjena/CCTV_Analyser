import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

STORE_DIR_PATTERN = re.compile(r"^Store\s*(\d+)$", re.IGNORECASE)
CAMERA_FILE_PATTERN = re.compile(r"CAM\s*(\d+)", re.IGNORECASE)
LAYOUT_FILE_PATTERN = re.compile(r".*layout.*\.(png|jpg|jpeg)$", re.IGNORECASE)

@dataclass
class StoreAssets:
    store_id: str
    store_name: str
    root_path: Path
    camera_files: Dict[str, Path]
    layout_image: Optional[Path] = None


def _store_root_for_path(path: Path, repo_root: Path) -> Optional[Path]:
    relative_parts = path.relative_to(repo_root).parts
    for part in relative_parts:
        if STORE_DIR_PATTERN.match(part):
            return repo_root / part
    return None


def _extract_camera_id(file_name: str) -> Optional[str]:
    match = CAMERA_FILE_PATTERN.search(file_name)
    if not match:
        return None
    return f"CAM_{int(match.group(1))}"


def discover_store_catalog(repository_root: Path) -> Dict[str, StoreAssets]:
    assets: Dict[str, StoreAssets] = {}

    for candidate_path in repository_root.rglob("*"):
        if not candidate_path.is_file():
            continue

        store_root = _store_root_for_path(candidate_path, repository_root)
        if store_root is None:
            continue

        top_store_folder = store_root.name
        store_id_match = STORE_DIR_PATTERN.match(top_store_folder)
        if not store_id_match:
            continue

        store_id = f"STORE_{store_id_match.group(1)}"
        if store_id not in assets:
            assets[store_id] = StoreAssets(
                store_id=store_id,
                store_name=f"{top_store_folder}",
                root_path=store_root,
                camera_files={},
                layout_image=None,
            )

        store_asset = assets[store_id]

        if candidate_path.suffix.lower() == ".mp4":
            camera_key = _extract_camera_id(candidate_path.name)
            if camera_key:
                store_asset.camera_files[camera_key] = candidate_path
        elif LAYOUT_FILE_PATTERN.match(candidate_path.name):
            store_asset.layout_image = candidate_path

    return assets

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from app.dataset import discover_store_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSACTION_CSV = REPO_ROOT / "POS - sample transactionsb1e826f.csv"

@dataclass(frozen=True)
class StoreDefinition:
    store_id: str
    name: str
    transaction_csv_path: Optional[Path]
    video_root: Optional[Path]
    layout_image_path: Optional[Path]


def load_store_definitions() -> Dict[str, StoreDefinition]:
    definitions: Dict[str, StoreDefinition] = {}

    if DEFAULT_TRANSACTION_CSV.exists():
        definitions["ST1008"] = StoreDefinition(
            store_id="ST1008",
            name="ST1008 - Brigade Road",
            transaction_csv_path=DEFAULT_TRANSACTION_CSV,
            video_root=None,
            layout_image_path=None,
        )

    discovered_stores = discover_store_catalog(REPO_ROOT)
    for store_id, assets in discovered_stores.items():
        definitions[store_id] = StoreDefinition(
            store_id=store_id,
            name=assets.store_name,
            transaction_csv_path=None,
            video_root=assets.root_path,
            layout_image_path=assets.layout_image,
        )

    return definitions

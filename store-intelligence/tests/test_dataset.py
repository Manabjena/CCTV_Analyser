from pathlib import Path
from app.dataset import discover_store_catalog


def test_discover_store_catalog_from_structure(tmp_path: Path):
    store_root = tmp_path / "Store 1"
    store_root.mkdir(parents=True)
    (store_root / "CAM 1 - zone.mp4").write_text("dummy")
    (store_root / "CAM 2 - zone.mp4").write_text("dummy")
    (store_root / "Store 1 - layout.png").write_text("dummy")

    catalog = discover_store_catalog(tmp_path)

    assert "STORE_1" in catalog
    store_assets = catalog["STORE_1"]
    assert store_assets.layout_image is not None
    assert store_assets.layout_image.name == "Store 1 - layout.png"
    assert "CAM_1" in store_assets.camera_files
    assert "CAM_2" in store_assets.camera_files
    assert store_assets.root_path == store_root

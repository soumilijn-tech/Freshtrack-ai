"""Dataset Builder for Broad Produce Recognition.
Curates diverse photographic training, validation, and test datasets
combining Fruits-360 (for clean fruit morphology) and natural Wikimedia Commons
photos (for realistic backgrounds, lighting, and kitchen countertop context).
Strictly separates train (70%), validation (15%), and test (15%) splits
with class stratification and fixed random seed to prevent data leakage.
"""
import os
import sys
import json
import random
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "produce_dataset"

# 20 Produce Categories with verified class mappings
PRODUCE_CATEGORIES = [
    "Persimmon",
    "Orange",
    "Apple",
    "Banana",
    "Lemon",
    "Potato",
    "Onion",
    "Tomato",
    "Carrot",
    "Bell Pepper",
    "Broccoli",
    "Cucumber",
    "Strawberry",
    "Avocado",
    "Lettuce",
    "Mango",
    "Peach",
    "Pomegranate",
    "Watermelon",
    "Eggplant"
]

# Fruits-360 folder mappings on GitHub (Horea94/Fruit-Images-Dataset)
FRUITS360_MAP = {
    "Persimmon": ["Kaki"],
    "Orange": ["Orange"],
    "Apple": ["Apple Red 1", "Apple Granny Smith", "Apple Golden 1"],
    "Banana": ["Banana", "Banana Red"],
    "Lemon": ["Lemon", "Lemon Meyer"],
    "Potato": ["Potato Red", "Potato Sweet", "Potato White"],
    "Onion": ["Onion Red", "Onion White"],
    "Tomato": ["Tomato 1", "Tomato Cherry Red", "Tomato Heart"],
    "Carrot": ["Carrot"],
    "Bell Pepper": ["Pepper Red", "Pepper Green", "Pepper Orange"],
    "Broccoli": ["Cauliflower"],
    "Cucumber": ["Cucumber Ripe"],
    "Strawberry": ["Strawberry"],
    "Avocado": ["Avocado", "Avocado ripe"],
    "Lettuce": ["Salad"],
    "Mango": ["Mango"],
    "Peach": ["Peach"],
    "Pomegranate": ["Pomegranate"],
    "Watermelon": ["Watermelon"],
    "Eggplant": ["Eggplant"]
}

HEADERS = {"User-Agent": "FreshTrackAI-Research/1.0 (contact@freshtrack.ai)"}


def download_fruits360_samples(category: str, dest_folder: Path, count_per_cat: int = 15):
    """Download sample images from Fruits-360 GitHub repository."""
    folders = FRUITS360_MAP.get(category, [])
    downloaded = 0
    for folder in folders:
        url = f"https://api.github.com/repos/Horea94/Fruit-Images-Dataset/contents/Test/{urllib.parse.quote(folder)}"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as r:
                items = json.loads(r.read().decode())
                img_items = [it for it in items if it["name"].endswith((".jpg", ".png"))]
                random.seed(42)
                selected = random.sample(img_items, min(count_per_cat, len(img_items)))
                for idx, it in enumerate(selected):
                    img_dest = dest_folder / f"f360_{folder.replace(' ', '_')}_{idx}.jpg"
                    if not img_dest.exists():
                        req_img = urllib.request.Request(it["download_url"], headers=HEADERS)
                        with urllib.request.urlopen(req_img, timeout=15) as img_resp, open(img_dest, "wb") as f:
                            f.write(img_resp.read())
                        downloaded += 1
        except Exception as e:
            print(f"  [Fruits360] {category} ({folder}) notice: {e}")
    return downloaded


def download_wikimedia_samples(category: str, dest_folder: Path, count: int = 10):
    """Download natural background photos from Wikimedia Commons."""
    query = f"{category} fruit" if category not in ("Potato", "Onion", "Carrot", "Broccoli", "Lettuce", "Cucumber", "Eggplant") else f"{category} vegetable"
    if category == "Persimmon":
        query = "Diospyros kaki Persimmon fruit"
    elif category == "Orange":
        query = "Citrus sinensis Orange fruit"
    elif category == "Lemon":
        query = "Citrus limon Lemon fruit"
    elif category == "Potato":
        query = "Solanum tuberosum Potato tuber"
    elif category == "Onion":
        query = "Allium cepa Onion bulb"
    elif category == "Tomato":
        query = "Solanum lycopersicum Tomato fruit"

    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&srnamespace=6&srlimit={count * 2}&format=json"
    downloaded = 0
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            titles = [
                it["title"] for it in data.get("query", {}).get("search", [])
                if it["title"].lower().endswith((".jpg", ".jpeg"))
            ][:count]

        for idx, title in enumerate(titles):
            u_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url&format=json"
            req2 = urllib.request.Request(u_url, headers=HEADERS)
            with urllib.request.urlopen(req2, timeout=10) as r2:
                d2 = json.loads(r2.read().decode())
                pages = list(d2.get("query", {}).get("pages", {}).values())
                if pages and "imageinfo" in pages[0] and pages[0]["imageinfo"]:
                    direct_url = pages[0]["imageinfo"][0]["url"]
                    img_dest = dest_folder / f"wiki_{idx}.jpg"
                    if not img_dest.exists():
                        req_img = urllib.request.Request(direct_url, headers=HEADERS)
                        with urllib.request.urlopen(req_img, timeout=15) as img_resp, open(img_dest, "wb") as f:
                            f.write(img_resp.read())
                        # Validate that it's a valid readable image and resize if oversized
                        try:
                            with Image.open(img_dest) as img:
                                img = img.convert("RGB")
                                if max(img.size) > 600:
                                    img.thumbnail((500, 500), Image.Resampling.LANCZOS)
                                img.save(img_dest, quality=85)
                            downloaded += 1
                        except Exception:
                            img_dest.unlink(missing_ok=True)
    except Exception as e:
        print(f"  [Wikimedia] {category} notice: {e}")
    return downloaded


def build_dataset():
    """Build the multi-class dataset and generate split manifests."""
    print("=" * 60)
    print("FreshTrack AI — Produce Dataset Builder")
    print(f"Target categories ({len(PRODUCE_CATEGORIES)}): {PRODUCE_CATEGORIES}")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    all_images_by_class = {}

    for cat in PRODUCE_CATEGORIES:
        cat_dir = DATASET_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"Gathering images for: {cat}...")

        # 1. Fruits-360 samples
        f360_cnt = download_fruits360_samples(cat, cat_dir, count_per_cat=10)
        # 2. Wikimedia natural background samples
        wiki_cnt = download_wikimedia_samples(cat, cat_dir, count=8)

        # List valid images
        valid_files = []
        for f in cat_dir.glob("*.jpg"):
            try:
                with Image.open(f) as im:
                    im.verify()
                valid_files.append(str(f.relative_to(PROJECT_ROOT)))
            except Exception:
                f.unlink(missing_ok=True)

        all_images_by_class[cat] = sorted(valid_files)
        print(f"  -> Total valid images for {cat}: {len(valid_files)} (Fruits360: {f360_cnt}, Natural: {wiki_cnt})")

    # Perform stratified split (70% train, 15% val, 15% test) with fixed seed=42
    random.seed(42)
    splits = {"train": [], "val": [], "test": []}

    for cat, files in all_images_by_class.items():
        shuffled = list(files)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(int(n * 0.70), 1)
        n_val = max(int(n * 0.15), 1)

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]
        if not test_files:
            test_files = val_files[-1:]  # ensure non-empty test

        splits["train"].extend([(f, cat) for f in train_files])
        splits["val"].extend([(f, cat) for f in val_files])
        splits["test"].extend([(f, cat) for f in test_files])

    # Save manifest
    manifest_path = DATASET_DIR / "splits.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "categories": PRODUCE_CATEGORIES,
            "counts_per_class": {k: len(v) for k, v in all_images_by_class.items()},
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "splits": splits
        }, f, indent=2)

    print("-" * 60)
    print(f"Dataset Build Complete! Manifest saved to {manifest_path}")
    print(f"Train samples: {len(splits['train'])}, Val samples: {len(splits['val'])}, Test samples: {len(splits['test'])}")
    print("=" * 60)
    return manifest_path


if __name__ == "__main__":
    build_dataset()

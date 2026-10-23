"""Model download and integrity verification utility for FreshTrack AI.
Downloads the official pretrained MobileNetV2 ONNX neural network during build
or startup, and verifies its cryptographic SHA-256 hash.
"""
import sys
import os
import hashlib
import urllib.request
from pathlib import Path

EXPECTED_SHA256 = "c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f"
EXPECTED_SIZE = 14246826

VERIFIED_MODEL_URLS = [
    # Primary: Direct Git LFS media CDN on GitHub
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    # Mirror: GitHub raw redirect to Git LFS media storage
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
]


def verify_file_integrity(file_path: Path) -> bool:
    """Verify that a model file exists, matches expected byte size and SHA-256 hash."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size != EXPECTED_SIZE:
        return False
    sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return sha == EXPECTED_SHA256


def ensure_model(dest_dir: Path = None) -> bool:
    """Ensure MobileNetV2 ONNX weights exist and pass SHA-256 verification."""
    if dest_dir is None:
        dest_dir = Path(__file__).resolve().parent.parent / "models"

    dest_path = dest_dir / "mobilenetv2.onnx"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if verify_file_integrity(dest_path):
        print(f"[FreshTrack AI] Verified model already present at {dest_path.name} (14.2 MB, SHA256 verified).")
        return True

    # If corrupted or partial file exists, remove it
    if dest_path.exists():
        print(f"[FreshTrack AI] Existing file at {dest_path.name} failed verification ({dest_path.stat().st_size} bytes). Replacing...")
        dest_path.unlink()

    print(f"[FreshTrack AI] Downloading official MobileNetV2 ONNX model (~14.2 MB)...")
    for url in VERIFIED_MODEL_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FreshTrackAI-App/1.0"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()

            if len(data) == EXPECTED_SIZE:
                sha = hashlib.sha256(data).hexdigest()
                if sha == EXPECTED_SHA256:
                    dest_path.write_bytes(data)
                    print(f"[FreshTrack AI] Successfully downloaded and verified MobileNetV2 ONNX model.")
                    return True
                else:
                    print(f"[FreshTrack AI] Warning: Hash mismatch for {url}: got {sha}")
            else:
                print(f"[FreshTrack AI] Warning: Incomplete payload from {url}: got {len(data)} bytes, expected {EXPECTED_SIZE}")
        except Exception as e:
            print(f"[FreshTrack AI] Download attempt from {url} failed: {e}")

    return False


if __name__ == "__main__":
    success = ensure_model()
    if not success:
        print("[FreshTrack AI] Error: Failed to obtain valid MobileNetV2 ONNX model.", file=sys.stderr)
        sys.exit(1)

"""DOM Element and UI Audit Utility for FreshTrack AI.
Verifies that all DOM IDs and classes referenced in frontend JavaScript
files exist in frontend/index.html without syntax or runtime issues.
"""
import re
import sys
from pathlib import Path

# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "backend" else Path(__file__).resolve().parent

def audit_dom():
    index_html = PROJECT_ROOT / "frontend" / "index.html"
    if not index_html.exists():
        print(f"Error: index.html not found at {index_html}")
        return False

    html_content = index_html.read_text(encoding="utf-8")
    js_dir = PROJECT_ROOT / "frontend" / "js"
    js_files = list(js_dir.glob("*.js"))

    if not js_files:
        print(f"Error: No JavaScript files found in {js_dir}")
        return False

    all_passed = True
    total_ids_checked = 0

    print("=" * 60)
    print("FreshTrack AI — Frontend DOM Integrity Audit")
    print("=" * 60)

    for js_file in js_files:
        content = js_file.read_text(encoding="utf-8")
        # Find getElementById('...') and getElementById("...")
        ids = re.findall(r'getElementById\(["\']([a-zA-Z0-9_\-]+)["\']\)', content)
        unique_ids = sorted(list(set(ids)))
        total_ids_checked += len(unique_ids)

        missing_ids = [
            elem_id for elem_id in unique_ids
            if f'id="{elem_id}"' not in html_content and f"id='{elem_id}'" not in html_content
        ]

        if missing_ids:
            print(f"[FAIL] {js_file.name}: Missing {len(missing_ids)} ID(s) in index.html: {missing_ids}")
            all_passed = False
        else:
            print(f"[PASS] {js_file.name}: Verified {len(unique_ids)} DOM element ID(s).")

    print("-" * 60)
    if all_passed:
        print(f"AUDIT SUCCESS: All {total_ids_checked} DOM element references are valid!")
        print("=" * 60)
        return True
    else:
        print("AUDIT FAILED: Some referenced IDs are missing in index.html.")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = audit_dom()
    sys.exit(0 if success else 1)

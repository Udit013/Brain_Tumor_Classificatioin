"""Export the API's OpenAPI schema to web/openapi.json (the frontend's type source).

    PYTHONPATH=src python scripts/export_openapi.py          # write
    PYTHONPATH=src python scripts/export_openapi.py --check  # CI: fail on drift

After writing, regenerate the TypeScript types with `npm --prefix web run gen:api`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TARGET = ROOT / "web" / "openapi.json"


def schema() -> str:
    from neuroclass.api.app import create_app

    return json.dumps(create_app(run_migrations=False).openapi(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    current = schema()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != current:
            print("web/openapi.json is out of date: run scripts/export_openapi.py and "
                  "`npm --prefix web run gen:api`", file=sys.stderr)
            return 1
        print("web/openapi.json matches the API")
        return 0
    TARGET.write_text(current)
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

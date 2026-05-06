from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppstructure_api import PPStructureEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPStructure on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    engine = PPStructureEngine()
    result = engine.predict_file(args.image)
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

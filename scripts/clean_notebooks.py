from __future__ import annotations

import json
import sys
from pathlib import Path


def clean_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks")
    for path in target.rglob("*.ipynb"):
        clean_notebook(path)


if __name__ == "__main__":
    main()

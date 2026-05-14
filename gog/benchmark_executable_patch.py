"""Run the public Vue executable patch benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from gog_cli.executable_patch_benchmark import main


if __name__ == "__main__":
    main()

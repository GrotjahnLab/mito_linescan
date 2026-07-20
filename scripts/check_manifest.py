#!/usr/bin/env python3
"""Manifest drift check: every runtime dependency declared in setup.py's
`install_requires` must also appear in environment.yml (conda deps or the pip
sub-list).

This guards against the DEP-2 class of bug, where a package the code imports is
listed in one manifest but missing from the other, so a fresh env built from
environment.yml crashes on import. Exits non-zero (failing CI) on any drift.

Stdlib only — no third-party parsing deps, so it runs before the env is built.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PY = REPO_ROOT / "setup.py"
ENV_YML = REPO_ROOT / "environment.yml"

# Packages that are intentionally optional (lazy-imported with a graceful skip)
# and therefore allowed to live in environment.yml without being a hard
# install_requires. Keep this list tiny and justified.
OPTIONAL_ALLOWED = {"localthickness"}


def _canon(name):
    """Lowercase and strip any version specifier / extras from a requirement."""
    name = name.strip().strip("\"'")
    name = re.split(r"[<>=!~;\[ ]", name, maxsplit=1)[0]
    return name.lower()


def install_requires():
    tree = ast.parse(SETUP_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "install_requires":
            return {_canon(x) for x in ast.literal_eval(node.value)}
    return set()


def env_packages():
    """Parse environment.yml dependencies without PyYAML (stdlib-only).

    Handles the two shapes used here: top-level `- pkg` conda deps and a
    `- pip:` block whose children are `- pkg` pip deps. Skips `python`, `pip`,
    and editable installs (`-e .`).
    """
    pkgs = set()
    in_deps = False
    for raw in ENV_YML.read_text().splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if re.match(r"^dependencies:", stripped):
            in_deps = True
            continue
        if in_deps and re.match(r"^[A-Za-z_]+:", stripped) and not stripped.startswith("pip:"):
            # A new top-level key ends the dependencies block.
            in_deps = False
        if not in_deps:
            continue
        if stripped in ("pip:", "- pip:"):
            continue
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item in ("pip:",) or item.startswith("-e") or item == ".":
                continue
            name = _canon(item)
            if name and name not in ("python", "pip"):
                pkgs.add(name)
    return pkgs


def main():
    req = install_requires()
    env = env_packages()
    missing = sorted(req - env - OPTIONAL_ALLOWED)

    print(f"setup.py install_requires ({len(req)}): {sorted(req)}")
    print(f"environment.yml packages ({len(env)}): {sorted(env)}")

    if missing:
        print()
        print("MANIFEST DRIFT — declared in setup.py but missing from "
              "environment.yml:")
        for m in missing:
            print(f"  - {m}")
        print("\nAdd them to environment.yml (or mark truly-optional deps in "
              "OPTIONAL_ALLOWED).")
        return 1

    print("\nOK: every install_requires package is present in environment.yml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

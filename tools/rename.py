"""Replace the provisional name everywhere, in one pass (release step).

    python -m tools.rename --package pinfall --display Pinfall --hf-id org/Pinfall-7M [--dry-run]

Rules, in order: the Hugging Face id `account/Nom-7M`; class prefixes `NomForecaster`, `NomModel`, `NomConfig`; the display
name `Nom`; identifiers `nom_...`; the package `nom`; environment variables `NOM_...`. Then `src/nom/` moves to `src/<package>/`. Text files only
(code, Markdown, YAML, TOML, CFF, CSV, JSON, NOTICE); figures are regenerated afterwards with `python -m showcase.build`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

TEXT = (".py", ".md", ".toml", ".cff", ".yaml", ".yml", ".csv", ".json", ".txt", ".cfg")
NAMED = ("NOTICE",)
SKIP_DIRS = {".git", "weights", "__pycache__", ".pytest_cache", "dist", "build"}
OLD_HF = "account/Nom-7M"
LEFTOVER = re.compile(r"\bnom\b|\bNom\b|\bnom_|\bNOM_|account/Nom")
SELF = {os.path.join("tools", "rename.py"), os.path.join("tests", "test_rename.py")}   # hold the old name on purpose


def apply(text: str, package: str, display: str, hf_id: str) -> str:
    text = text.replace(OLD_HF, hf_id)
    text = re.sub(r"\bNom(?=(Forecaster|Model|Config)\b)", display, text)
    text = re.sub(r"\bNom\b", display, text)
    text = re.sub(r"\bnom_", f"{package}_", text)
    text = re.sub(r"\bnom\b", package, text)
    text = re.sub(r"\bNOM_", f"{package.upper()}_", text)
    return text


def _files(root: str):
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".venv")]
        for n in names:
            p = os.path.join(d, n)
            if (n.endswith(TEXT) or n in NAMED) and os.path.relpath(p, root) not in SELF:
                yield p


def run(root: str, package: str, display: str, hf_id: str, dry_run: bool = False) -> list[str]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", package):
        raise ValueError(f"package name must be a lowercase Python identifier, got {package!r}")
    changed = []
    for path in _files(root):
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new = apply(old, package, display, hf_id)
        if new != old:
            changed.append(os.path.relpath(path, root))
            if not dry_run:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    src_old, src_new = os.path.join(root, "src", "nom"), os.path.join(root, "src", package)
    if os.path.isdir(src_old) and not dry_run:
        shutil.move(src_old, src_new)
    return changed


def leftovers(root: str) -> list[str]:
    """Lines still holding the provisional name."""
    pat = LEFTOVER
    out = []
    for path in _files(root):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if pat.search(line):
                    out.append(f"{os.path.relpath(path, root)}:{i}: {line.strip()[:100]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--display", required=True)
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    changed = run(root, a.package, a.display, a.hf_id, a.dry_run)
    print(f"{len(changed)} files {'would change' if a.dry_run else 'changed'}")
    for p in changed:
        print("  ", p)
    if not a.dry_run:
        left = leftovers(root)
        print(f"leftovers: {len(left)}")
        for x in left:
            print("  ", x)
        print("next: pip install -e . ; pytest ; python -m showcase.build ... (figures carry the name)")


if __name__ == "__main__":
    main()

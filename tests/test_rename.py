"""The one-pass rename (release step): on a copy of the repository, every occurrence of the provisional name changes and
the package still imports."""
import os
import shutil
import subprocess
import sys

from tools import rename

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_rules_on_known_text():
    text = ('import nom\nfrom nom import NomForecaster, NomModel\nf = nom_forecast_fn(x)  # Nom (ours)\n'
            'NomForecaster.from_pretrained("account/Nom-7M")\nname = "nom"\nCopyright 2026 The Nom Authors\nnominal\n')
    out = rename.apply(text, package="pinfall", display="Pinfall", hf_id="org/Pinfall-7M")
    assert out == ('import pinfall\nfrom pinfall import PinfallForecaster, PinfallModel\nf = pinfall_forecast_fn(x)  '
                   '# Pinfall (ours)\nPinfallForecaster.from_pretrained("org/Pinfall-7M")\nname = "pinfall"\n'
                   'Copyright 2026 The Pinfall Authors\nnominal\n')


def test_rename_a_copy_of_the_repository(tmp_path):
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    dst = tmp_path / "repo"
    for f in files:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), dst / f)
    changed = rename.run(str(dst), package="pinfall", display="Pinfall", hf_id="org/Pinfall-7M")
    assert changed and (dst / "src" / "pinfall" / "forecaster.py").exists() and not (dst / "src" / "nom").exists()
    left = rename.leftovers(str(dst))
    assert left == [], left[:5]
    assert "account/Nom-7M" in (dst / "tools" / "rename.py").read_text()          # the tool itself is left alone
    code = "import sys; sys.path.insert(0, 'src'); import pinfall; assert hasattr(pinfall, 'PinfallForecaster')"
    subprocess.run([sys.executable, "-c", code], cwd=dst, check=True)


def test_upper_case_environment_variables():
    out = rename.apply('os.environ["NOM_WEIGHTS"]  # NOMINAL\n', package="pinfall", display="Pinfall",
                       hf_id="org/Pinfall-7M")
    assert out == 'os.environ["PINFALL_WEIGHTS"]  # NOMINAL\n'
    assert rename.LEFTOVER.search("NOM_WEIGHTS") and not rename.LEFTOVER.search("NOMINAL")

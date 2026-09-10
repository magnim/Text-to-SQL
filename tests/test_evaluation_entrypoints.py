from pathlib import Path
import subprocess
import sys

import pytest


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "evaluation/run_current_legacy.py",
        "evaluation/run_reset_ablation.py",
        "evaluation/run_unseen_and_heldout.py",
        "evaluation/run_v2_regression_suite.py",
    ],
)
def test_argument_driven_evaluation_scripts_support_direct_execution(script):
    result=subprocess.run(
        [sys.executable,script,"--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode==0, result.stderr


def test_zero_shot_evaluation_script_bootstraps_repo_root_for_direct_execution():
    script=ROOT/"evaluation/run_v2_zero_shot_acceptance.py"
    code=(
        "import pathlib,runpy,sys; "
        f"root=pathlib.Path({str(ROOT)!r}); "
        "sys.path=[str(root/'evaluation')]+[p for p in sys.path if p and pathlib.Path(p).resolve()!=root]; "
        f"runpy.run_path({str(script)!r},run_name='not_main')"
    )
    result=subprocess.run(
        [sys.executable,"-c",code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode==0, result.stderr

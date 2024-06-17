import pathlib
import matplotlib
import runpy
import pytest

matplotlib.use("Agg")

scripts_to_test = [
    "dipole_forces.py",
    "psf_example.py",
    "trapping_fields.py",
    "trapping_forces.py",
]
script_path = pathlib.Path(__file__, "../../../", "examples")
scripts = [script_path / script for script in scripts_to_test]


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
@pytest.mark.parametrize("script", scripts)
def test_script_execution(script):
    runpy.run_path(script)

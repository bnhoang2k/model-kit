"""The four measured trees under test/fixtures/qsm: where they are, and what they are. No tests."""

from pathlib import Path

QSM_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "qsm"

# fixture file -> the TreeQSM file it was converted from (Demol et al. 2021)
QSM_SOURCES = {
    "FSYL_15.skeleton.npz": "FSYL15.mat",  # beech, the set's shortest (18.0 m)
    "FSYL_13.skeleton.npz": "FSYL13.mat",  # beech, the set's tallest (23.8 m)
    "FEXC_5.skeleton.npz": "FEXC5.mat",  # ash, the set's shortest (12.9 m)
    "FEXC_14.skeleton.npz": "FEXC14.mat",  # ash, the set's tallest (22.7 m)
}
QSM_FILES = tuple(QSM_FIXTURES / name for name in QSM_SOURCES)

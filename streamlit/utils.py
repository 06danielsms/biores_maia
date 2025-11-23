"""Utilities shared by the Streamlit views."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FILES_DIR = REPO_ROOT / "test_files"
DEFAULT_SAMPLE = TEST_FILES_DIR / "sample_1_diabetes.txt"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def ensure_repo_on_path() -> None:
    """Allow importing modules that live at the repository root (e.g. scripts.clean_en)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.append(root)


@lru_cache
def load_default_text() -> str:
    """Read the default sample text bundled with the project."""
    if DEFAULT_SAMPLE.exists():
        return DEFAULT_SAMPLE.read_text(encoding="utf-8")
    text = """Participants aged <17 years received Pregabalin, orally, twice daily in equally divided doses, for the double-blind treatment phase of 12 weeks in following manner: 1) body weight >=30 kg: Pregabalin 10 mg/kg/day as capsule or oral solution (using oral solution of strength 20 mg/mL), up to a maximum of 600 mg/day; 2) body weight <30 kg: pregabalin 14 mg/kg/day as oral solution (using oral solution of strength 20 mg/mL), up to a maximum of 600 mg/day. Participants aged >=17 years received Pregabalin 600 mg/day, capsule or oral solution, orally twice daily in equally divided doses, for the double-blind treatment phase of 12 weeks.
The study is designed to evaluate the safety, tolerability and efficacy of two doses of pregabalin as add-on treatment in pediatric and adult subjects with Primary Generalized Tonic-Clonic (PGTC) seizures as compared to placebo. It is hypothesized that both doses of pregabalin will demonstrated superior efficacy when compared to placebo by reducing PGTC seizure frequency and that pregabalin will be safe and well tolerated.
A Safety, Efficacy and Tolerability Trial of Pregabalin as Add-On Treatment in Pediatric and Adult Subjects With Primary Generalized Tonic-Clonic (i.e., Grand Mal) Seizures.
Participants aged less than (<) 17 years received Pregabalin, orally, twice daily in equally divided doses, for the double-blind treatment phase of 12 weeks in following manner: 1) body weight greater than or equal to (>=)30 kg: Pregabalin 5 milligram per kilogram per day (mg/kg/day) as capsule or oral solution (using oral solution of strength 20 milligram per milliliter [mg/mL]), up to a maximum of 300 milligram per day (mg/day); 2) body weight <30 kg: pregabalin 7 mg/kg/day as oral solution (using oral solution of strength 20 mg/mL), up to a maximum of 300 mg/day. Participants aged >=17 years received Pregabalin 300 mg/day, capsule or oral solution, orally twice daily in equally divided doses, for the double-blind treatment phase of 12 weeks."""
    return text

def read_uploaded_text(upload) -> Optional[str]:
    """Read a Streamlit UploadedFile into a UTF-8 string."""
    if upload is None:
        return None
    try:
        return upload.getvalue().decode("utf-8")
    except Exception:
        return None

@lru_cache
def load_repo_config() -> Dict[str, Any]:
    """Load the main YAML config so Streamlit mirrors CLI defaults."""
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


ensure_repo_on_path()

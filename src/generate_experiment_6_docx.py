"""
Backward compatibility shim for generate_experiment_6_docx module.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.experiment_6.src.generate_experiment_6_docx import *

if __name__ == "__main__":
    generate_experiment_6_docx()

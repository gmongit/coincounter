"""Gemeinsame Konstanten für das ganze Projekt."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
DATA_OUT = ROOT / "data"  # generierte Daten (Crops etc.), nicht im Git

# Reihenfolge = YOLO-Klassen-ID in dataset2 (aus Einzelmünzbildern rekonstruiert, siehe README).
# Namen entsprechen den Ordnernamen in coins-dataset/classified.
CLASS_NAMES = ["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

# Wert in Cent pro Klasse
CLASS_VALUES_CENT = {"1c": 1, "2c": 2, "5c": 5, "10c": 10, "20c": 20, "50c": 50, "1e": 100, "2e": 200}

# Offizielle Durchmesser in mm (nützlich für relative Größen-Features)
CLASS_DIAMETER_MM = {"1c": 16.25, "2c": 18.75, "5c": 21.25, "10c": 19.75,
                     "20c": 22.25, "50c": 24.25, "1e": 23.25, "2e": 25.75}

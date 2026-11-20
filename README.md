# coincounter

Gruppenprojekt Image Processing: Gesamtwert einer Ansammlung von Euromünzen aus einem Foto bestimmen –
robust gegenüber unterschiedlichen Aufnahmebedingungen (Licht, Untergrund, Abstand).

## Setup

Python 3.13 (3.14+ wird von PyTorch evtl. noch nicht unterstützt).

```powershell
# Windows
py -3.13 -m venv .venv            # oder: <Pfad zu python3.13.exe> -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installiert die **CPU-Version** von PyTorch. Mit NVIDIA-GPU vorher die passende
CUDA-Version von torch/torchvision installieren (https://pytorch.org/get-started/locally/).

## Daten

Die Datensätze sind **nicht im Repo** (`datasets/` ist gitignored) und müssen manuell nach `datasets/` geholt werden:

| Ordner | Quelle | Lizenz |
|---|---|---|
| `datasets/coins-dataset` | https://github.com/kaa/coins-dataset (`git clone` nach `datasets/`) | MIT |
| `datasets/dataset2` | https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset (Download entpacken nach `datasets/dataset2/`, sodass `images/` und `labels/` direkt darin liegen) | Apache 2.0 |

Erwartete Struktur:

```
datasets/
├── coins-dataset/          # github: Einzelmünz-Crops, nach Klassen sortiert (Klassifikation)
│   └── classified/{train,test}/{1c,2c,5c,10c,20c,50c,1e,2e}/
└── dataset2/               # 150 Fotos, YOLO-Labels (Detection)
    ├── images/  labels/
    ├── data.yaml           # generiert
    └── train.txt val.txt   # generiert (fester Split, seed 42)
```

**Klassen-IDs in dataset2** (nicht mitgeliefert, aus Einzelmünzbildern rekonstruiert und per Crop-Kontrolle bestätigt):

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|----|----|----|----|-----|-----|-----|----|----|
|    | 1c | 2c | 5c | 10c | 20c | 50c | 1€ | 2€ |

Definiert in `coincounter/constants.py`.

### Daten vorbereiten

```powershell
python -m scripts.prepare_dataset2
```

Erzeugt `datasets/dataset2/data.yaml` + Train/Val-Split (80/20) und schneidet alle gelabelten Münzen nach
`data/crops/dataset2/{train,val}/<klasse>/` aus – gleiche Struktur wie `coins-dataset/classified`, sodass beide
Datensätze gemeinsam für einen Klassifikator verwendet werden können. Der Crop-Split folgt dem Bild-Split
(kein Leakage).

## Projektstruktur

```
coincounter/     # Python-Package (gemeinsamer Code, Konstanten)
scripts/         # ausführbare Scripts, Aufruf mit  python -m scripts.<name>
datasets/        # Rohdaten
data/            # generierte Daten (gitignored)
runs/            # Trainingsergebnisse (gitignored)
```

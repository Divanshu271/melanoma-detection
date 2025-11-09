# QML Melanoma Classification - HAM10000

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download HAM10000 dataset and place in `data/` directory:
   - Images: `data/ham10000/`
   - Metadata: `data/HAM10000_metadata.csv`

## Usage

### Train QML Model:

```bash
python main.py --model qml --epochs 50 --batch_size 32 --n_qubits 4 --n_layers 2
```

### Train Classical Baseline:

```bash
python main.py --model classical --epochs 50 --batch_size 32
```

## Key Features

- ✓ Patient-level splitting (no data leakage)
- ✓ Proper train/val/test separation
- ✓ QML implementation with PennyLane
- ✓ Classical baseline for comparison
- ✓ Comprehensive evaluation metrics

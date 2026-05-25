# ECG Arrhythmia Benchmark

Benchmarking three deep learning architectures for ECG arrhythmia classification on the MIT-BIH dataset.

## Task

Multi-class classification of heartbeat segments into 5 AAMI classes: Normal (N), LBBB (L), RBBB (R), PVC (V), APC (A).

## Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — 48 records, ~100k heartbeat segments, sampled at 360 Hz. Downloaded automatically via `wfdb`.

## Models

| Model | Params | Val F1 | Test F1 |
|---|---|---|---|
| 1D CNN | 232k | 0.9700 | 0.9691 |
| BiLSTM | 546k | 0.9599 | 0.9645 |
| CNN + Transformer | — | 0.9750 | 0.9713 |

## Results

| Model | N | L | R | V | A | Macro F1 |
|---|---|---|---|---|---|---|
| CNN | 0.9939 | 0.9934 | 0.9922 | 0.9736 | 0.8923 | 0.9691 |
| BiLSTM | 0.9930 | 0.9942 | 0.9931 | 0.9692 | 0.8731 | 0.9645 |
| Transformer | 0.9951 | 0.9979 | 0.9963 | 0.9862 | 0.8810 | 0.9713 |

Class A (APC) is the hardest across all models — visually similar to Normal beats, confusion is almost entirely A→N.

## Setup

```bash
git clone https://github.com/sidharthguptax/ecg-arrhythmia-benchmark.git
cd ecg-arrhythmia-benchmark
pip install -r requirements.txt
```

## Usage

```bash
# Preprocess
python src/utils/preprocessing.py

# Train
python -m src.train --model cnn
python -m src.train --model lstm
python -m src.train --model transformer
```

Then open `notebooks/03_evaluation.ipynb` for full evaluation.

## Structure

```
├── data/               # raw + processed (gitignored)
├── notebooks/          # EDA, training curves, evaluation
├── src/
│   ├── models/         # cnn.py, lstm.py, transformer.py
│   ├── utils/          # preprocessing, metrics, visualize
│   ├── dataset.py
│   └── train.py
├── results/            # checkpoints + figures (gitignored)
└── config.yaml
```
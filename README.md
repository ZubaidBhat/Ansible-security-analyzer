# Ansible Security Analyzer — Run Guide

This README focuses on how to install, run, and use the tool.

## 1) Install
- Python 3.9+
- `pip install -r requirements.txt`

## 2) Scan Playbooks
- Hybrid (ML + rules, recommended):
  - `python src/scan_ml.py samples/`
- Rules-only (fast baseline):
  - `python src/scan.py samples/`

You can pass a single file or a directory. Non‑zero exit code if issues are found.

## 3) Useful Options (scan_ml)
- Export JSON: `--json results.json`
- Generate chart: `--chart --chart-out analysis.png`
- Disable ML (rules-only): `--no-ml`
- Adjust ML confidence: `--confidence 0.8`
- Train before scanning: `--train`
- Quiet mode (for CI): `--quiet`

Examples:
- `python src/scan_ml.py samples/ --json results.json --chart`
- `python src/scan_ml.py playbook.yml --confidence 0.75`


## 4) Train / Update the Model
- `python src/train_model.py`
- Saves model to `models/security_smell_classifier.pkl`
- Saves metadata to `models/training_metadata.json`

## 5) Sample Output (CLI)
```
📄 File: samples/insecure_web_app.yml
  [1]  HTTP_WITHOUT_TLS
       Location: play[0].tasks[0]
       Fix: Use HTTPS instead of HTTP
```

## 6) Quick Start (TL;DR)
```
pip install -r requirements.txt
python src/scan_ml.py samples/ --json results.json --chart
```

# AI-Powered Logistics Ingestion & Validation System

Fine-tuning Flan-T5-Large for automated structured data extraction from unstructured logistics communications.

---

## Overview

This project fine-tunes a large language model to extract structured data from messy logistics text (emails, SMS, warehouse messages). The system converts informal communications into database-ready JSON records with automated validation.

**Key Achievement:** 94% field-level accuracy, 100% JSON validity, zero catastrophic failures.

---

## Results

| Metric | Score |
|--------|-------|
| Field-Level Accuracy | 94% |
| JSON Parse Success | 100% |
| Perfect Extractions | 67.5% |
| Major Errors | 0% |
| Validation Loss | 0.0255 |

**Training:** 3 hyperparameter configurations tested, 27 minutes total on T4 GPU

**Improvement:** 3.76× better than baseline (zero-shot)

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Notebook
Open `Logistics_AI_FineTuning.ipynb` in Google Colab with T4 GPU enabled.

**Cells:**
1. Generate dataset (2,000 samples)
2. Fine-tune model (3 experiments)
3. Run inference demo
4. Analyze errors (200 test samples)
5. Create visualizations

**Runtime:** ~2 hours total

---

## Technical Details

**Model:** google/flan-t5-large (780M parameters)

**Fine-Tuning:** LoRA (rank=16, alpha=32, ~1% trainable params)

**Dataset:** 2,000 synthetic logistics messages with realistic noise (typos, abbreviations, informal language)

**Hardware:** Google Colab T4 GPU (16GB VRAM)

---

## Key Findings

**Data Quality Analysis:** Discovered that 24% of destination errors stemmed from data generation inconsistencies rather than model failures. The model correctly avoided hallucinating missing information, demonstrating conservative behavior appropriate for production systems.

**Hallucination Prevention:** Strict input-output consistency in training data successfully taught the model to output `null` for missing fields rather than fabricating values.

---

## Repository Contents

- `Logistics_AI_FineTuning.ipynb` - Complete workflow notebook
- `requirements.txt` - Python dependencies
- `Technical_Report.pdf` - Detailed documentation
- `README.md` - This file

---

## Requirements

```
transformers==4.44.0
peft==0.12.0
datasets==2.20.0
torch>=2.4.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

**Hardware:** GPU with 12GB+ VRAM recommended

---

## License

Academic project for educational purposes.

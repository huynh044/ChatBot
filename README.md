# ChatBot — Sentiment & Intent Chatbot

## Link Project
```bash
https://drive.google.com/drive/folders/10Bbm9mJLAectCVjfRU5YLkdLKMcu5tuZ?usp=sharing
```

## Overview
This project provides an ontology-aware chatbot for sentiment and intent detection with a focus on Vietnamese (and some English). The system emphasizes correct handling of negation and intensity. It includes two runnable pipelines (Classic with NLTK + Keras/TensorFlow, and Transformers with BERT), a simple Tkinter GUI, pre-trained models for immediate inference, and scripts for training/evaluation.

## Features
- Ontology-aware sentiment: detects negators and intensifiers to adjust polarity during inference.
- Two pipelines:
  - Classic: NLTK preprocessing with Keras/TensorFlow (.h5 models included).
  - Transformers: BERT-based inference and training scripts.
- Ready-to-run: pre-trained models and intent datasets.
- GUI demo: Tkinter-based chat interface.
- Vietnamese-first datasets: intent files can include a weight field expressing sentiment strength.

## Repository Structure (high-level)
The repository contains a nested folder `ChatBot/...`. Keep this structure unless you refactor imports and paths.

```
ChatBot/
└─ ChatBot/
   ├─ Code/                      # Classic pipeline (NLTK + Keras/TensorFlow)
   │  ├─ GUI.py                  # Tkinter UI demo
   │  ├─ RunProgram.py           # Inference (older)
   │  ├─ RunProgramv3.py         # Inference (v3)
   │  ├─ RunProgramv4.py         # Inference (v4, preferred)
   │  ├─ Training_Bot.py         # Training script (classic)
   │  ├─ weight.py, trongso.py   # Polarity weighting helpers
   │  └─ Model/
   │     ├─ OWLChatModel_v3.h5
   │     └─ OWLChatModel_v4.h5
   │
   ├─ Code_Transformers/         # BERT-based pipeline
   │  ├─ RunProgram_Transformers.py
   │  ├─ RunProgram_Transformers_v2.py
   │  ├─ RunProgram_Transformers_v4.py
   │  ├─ Training_Bot_Transformers*.py
   │  └─ Model/BertModel/        # tokenizer_config.json, special_tokens_map.json, etc.
   │
   ├─ Data/
   │  ├─ intent/                 # Intent JSONs (VN/EN), can include weight
   │  │  ├─ 01.json, 02.json, 03.json, Demo.json, TranfomerVN.json, ...
   │  ├─ classes.pkl, words.pkl, negation_words.pkl
   │  └─ classes.csv
   │
   ├─ Model/                     # Additional pre-trained models
   │  ├─ OWLChatModel.h5
   │  ├─ OWLChatModel_v2.h5
   │  ├─ ACCLE_model.h5
   │  └─ chatbot_model.h5
   │
   └─ Ontology/
      ├─ N_P_O_ver12.owl
      ├─ N_P_Ontology_Ver2.owl
      └─ N_P_ontology_ver1.owl
```

## Quick Start

### 1) Environment
- Python 3.9–3.11 (3.10 recommended)
- Windows/macOS/Linux (ensure Tkinter is available on your Python build)

Create and activate a virtual environment, then install dependencies:
```bash
# Core dependencies
pip install numpy nltk tensorflow keras scikit-learn pillow matplotlib

# Ontology & Transformers (optional but recommended)
pip install owlready2 transformers torch
```

Download NLTK resources if needed:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

### 2) Run the Classic Pipeline (GUI or Console)
Run from the repository root (the outer `ChatBot/`):
```bash
# GUI demo
python ChatBot/ChatBot/Code/GUI.py

# Console (preferred v4)
python ChatBot/ChatBot/Code/RunProgramv4.py
```
If paths are not found, ensure you are executing from the project root so relative paths resolve correctly.

### 3) Run the Transformers Pipeline
```bash
python ChatBot/ChatBot/Code_Transformers/RunProgram_Transformers_v4.py
# or
python ChatBot/ChatBot/Code_Transformers/RunProgram_Transformers_v2.py
```

## Data and Intent Schema
Intent files are in `ChatBot/ChatBot/Data/intent/` and follow this schema:
```json
{
  "intents": [
    {
      "tag": "negative_1",
      "patterns": ["sample text 1", "..."],
      "responses": ["a response"],
      "context": [],
      "weight": -1
    }
  ]
}
```
Notes:
- The `weight` field is optional and can encode sentiment strength (negative/positive scaling).
- Many datasets are Vietnamese-first (e.g., `TranfomerVN.json` with tags expressing negative/positive levels).

## Ontology (OWL)
Ontologies describe sentence structure and sentiment modifiers, including classes for sentence, subject, predicate, negation, negative adverbs, intensifiers/negators, emotion, and sentiment analysis.

Loading with `owlready2`:
```python
from owlready2 import get_ontology
onto = get_ontology("ChatBot/ChatBot/Ontology/N_P_O_ver12.owl").load()
for cls in list(onto.classes())[:15]:
    print(cls)
```

## Training and Evaluation

Classic training:
```bash
python ChatBot/ChatBot/Code/Training_Bot.py
```
Adjust dataset and output paths at the top of the script if necessary.

Transformers training:
```bash
python ChatBot/ChatBot/Code_Transformers/Training_Bot_Transformers.py
# or a versioned script, e.g. Training_Bot_Transformers_v3.py
```

Accuracy/F1:
```bash
python ChatBot/ChatBot/Code_Transformers/ACC_F1.py
```

## GUI
`Code/GUI.py` uses Tkinter and Pillow. Ensure Tkinter is available on your system. On some Linux environments, you may need to install system packages (e.g., `sudo apt-get install python3-tk`).

## Configuration Tips
- Refine intents: add examples to `patterns`, expand `responses`, and adjust `weight` for finer sentiment granularity.
- Extend ontology: add more Vietnamese negators/intensifiers to improve polarity handling.
- Model selection: Classic is lightweight and fast to demo; Transformers offer higher quality, preferably with a GPU.

## Dependencies (summary)
- Core: numpy, nltk, keras, tensorflow, scikit-learn
- GUI/Media: tkinter, Pillow, matplotlib
- Ontology: owlready2
- Transformers: transformers, torch

## Example Workflow for Negation and Intensity
1. Tokenize and normalize text (lowercase, optional stemming).
2. Match tokens against ontology classes for negators and intensifiers.
3. Adjust feature weights or logits before the final prediction.
4. Map predicted label to a response, optionally conditioned on the `weight` field.

## Troubleshooting
- NLTK resource errors (e.g., punkt): run the NLTK downloads above.
- Missing Tkinter: install a Python build that includes Tkinter or add system packages.
- FileNotFoundError for models/intents: run from the project root to preserve relative paths.
- GPU/TensorFlow issues: use CPU TensorFlow if GPU is unavailable; Transformers also run on CPU (slower).

## License
No license file was found. Consider adding a LICENSE (MIT/Apache-2.0) or marking as proprietary.

## Acknowledgements
- NLP/ML stack: NLTK, Keras/TensorFlow, Transformers.
- Ontology tooling: owlready2.
- Vietnamese intent and sentiment datasets curated with negation/intensity patterns.

- Issues and suggestions: open a GitHub issue or contact the maintainer.



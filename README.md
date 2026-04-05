
---
title: NLP Intelligence
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NLP Intelligence — Social Monitoring Web Application

Hexagonal (Ports & Adapters) architecture for Mongolian social media content analysis.

## Repository Structure

```
NLP-intelligence/
├── nlp_core/              # Domain Core — NER, sentiment, topic modeling, preprocessing (pure Python)
├── adapters/
│   ├── api/               # FastAPI REST adapter (routers, schemas, services)
│   ├── ner_mongolian/     # Fine-tuned NER model config/tokenizer (weights on HF Hub)
│   └── sumbee/            # Future Sumbee.mn integration
├── frontend/              # Next.js dashboard & admin panel
├── Data/                  # Training data & reference datasets (NOT used at runtime)
│   ├── data/              # CoNLL-format training/validation/test files (v1 pipeline)
│   ├── datav2/            # JSONL character-offset training data + scripts (v2 pipeline)
│   └── NER-dataset/       # Reference data (locations.json, abbreviations, names)
├── eval/                  # Model evaluation scripts
├── Dockerfile             # Multi-stage production build
├── nginx.conf             # Reverse proxy config (port 7860)
├── start.sh               # Docker entrypoint
└── requirements.txt
```

**Production code:** `nlp_core/`, `adapters/api/`, `frontend/` — included in Docker image.
**ML development:** `Data/`, `eval/` — excluded from Docker. See [Data/README.md](Data/README.md) for details.

## Model

The NER model is hosted on HuggingFace Hub: `Nomio4640/ner-mongolian`. It is downloaded automatically during Docker build and at runtime (if not cached locally). Model weights are NOT stored in git.

To version a new model after training:
```bash
git tag model-v1.0 -m "F1: 0.XX, trained on train_final.conll"
```

## Quick Start

### Local Development

```bash
# Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd adapters/api
PYTHONPATH=../../ uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

```bash
# Frontend
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:3000

### Docker

```bash
docker build -t nlp-intelligence .
docker run -p 7860:7860 nlp-intelligence
```

App: http://localhost:7860

### Usage

1. Open http://localhost:3000
2. Upload a CSV file with a `text` or `Text` column
3. View NER, sentiment, and network analysis results
4. Go to `/admin` to manage the knowledge base, labels, and stopwords

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/upload | Upload CSV for analysis |
| POST | /api/analyze | Analyze single text |
| POST | /api/analyze/batch | Analyze batch of texts |
| POST | /api/network | Get network graph data |
| POST | /api/insights | Get analysis insights |
| GET/POST | /api/admin/knowledge | Knowledge base CRUD |
| GET/POST | /api/admin/labels | Custom label mapping |
| GET/POST/DELETE | /api/admin/stopwords | Stopword management |

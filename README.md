# NLP Intelligence — Social Monitoring Web Application

Hexagonal (Ports & Adapters) architecture for Mongolian social media content analysis.

## Architecture

```
webapp/
├── nlp_core/         # Domain Core (pure Python, no framework deps)
├── adapters/api/     # FastAPI REST adapter
├── adapters/sumbee/  # Future Sumbee.mn integration
└── frontend/         # Next.js dashboard & admin
```

## Quick Start

### 1. Backend (FastAPI)

```bash
cd webapp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

#venv activate 
source /home/nomio/Documents/БСА/NLP-intelligence/venv/bin/activate 
# Start the API
cd adapters/api
PYTHONPATH=../../ uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 2. Frontend (Next.js)

```bash
cd webapp/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Dashboard: http://localhost:3000

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

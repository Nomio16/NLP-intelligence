# Sumbee.mn Integration Adapter

This directory is reserved for the future Sumbee.mn integration adapter.

## How to integrate with Sumbee

The `nlp_core` package is framework-independent. To integrate it into Sumbee's infrastructure:

### Option 1: Import as a Python Package
```python
from nlp_core import Preprocessor, NEREngine, SentimentAnalyzer, TopicModeler

preprocessor = Preprocessor()
ner = NEREngine()

text = "Монгол улс Оростой хэлэлцээ хийв"
clean = preprocessor.preprocess(text)
entities = ner.recognize(clean)
```

### Option 2: Call the FastAPI API
```
POST http://your-nlp-server:8000/api/analyze
Content-Type: application/json

{"text": "Монгол улс Оростой хэлэлцээ хийв"}
```

### Option 3: Build a custom Sumbee adapter
Create a `sumbee_adapter.py` that:
1. Listens to Sumbee's message queue (e.g., Kafka, RabbitMQ)
2. Processes incoming social posts through `nlp_core`
3. Writes results back to Sumbee's database

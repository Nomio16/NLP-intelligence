# NLP Intelligence — Deep Dive Guide

> A 2-3 hour reading guide for a CS student who knows basic Python, JavaScript, and has introductory ML knowledge. Written to explain this **specific project** — every concept is explained through what the code actually does.

---

## Table of Contents

1. [What This Project Actually Is](#1-what-this-project-actually-is)
2. [The Big Picture — How the Pieces Fit Together](#2-the-big-picture)
3. [Layer 1: The Frontend (Next.js + React)](#3-layer-1-frontend)
4. [Layer 2: The Network Layer (HTTP, APIs, Proxying)](#4-layer-2-network)
5. [Layer 3: The Backend API (FastAPI + Uvicorn)](#5-layer-3-backend)
6. [Layer 4: Text Preprocessing (The Unsung Hero)](#6-layer-4-preprocessing)
7. [Layer 5: NLP Models — The Brain](#7-layer-5-nlp-models)
8. [Layer 6: Data Persistence (SQLite)](#8-layer-6-database)
9. [How a Single Request Flows Through Everything](#9-full-request-flow)
10. [The ECONNRESET Bug — What Actually Happened](#10-econnreset-explained)
11. [Architecture Diagram](#11-architecture-diagram)
12. [What You Don't Know Yet (Knowledge Gaps)](#12-knowledge-gaps)
13. [Improvement Plan](#13-improvement-plan)
14. [Study Resources](#14-study-resources)

---

## 1. What This Project Actually Is

NLP Intelligence is a **social media monitoring tool** for Mongolian text. Users upload CSV files (e.g. Facebook comments, news articles, tweets) and the system automatically:

- **Reads** the sentiment of each text (positive / neutral / negative)
- **Finds** named entities — people (PER), organizations (ORG), locations (LOC)
- **Groups** texts into topics ("What are people talking about?")
- **Maps** relationships between entities ("Who is mentioned together with whom?")
- **Saves** results to a database for historical comparison

This is valuable because Mongolian is a **low-resource language** — there aren't many tools that do NLP for it. Most NLP tooling is built for English/Chinese.

---

## 2. The Big Picture — How the Pieces Fit Together

```
┌─────────────────────────────────────────────────────────┐
│                     YOUR COMPUTER                       │
│                                                         │
│  ┌──────────────┐    HTTP     ┌───────────────────────┐ │
│  │   FRONTEND   │◄──────────►│       BACKEND          │ │
│  │              │  requests   │                       │ │
│  │  Next.js     │  port 3000  │  FastAPI + Uvicorn    │ │
│  │  React       │  ────────►  │  port 8000            │ │
│  │  TypeScript  │  responses  │                       │ │
│  │              │  ◄────────  │  ┌─────────────────┐  │ │
│  │  browser tab │             │  │  NLP CORE       │  │ │
│  │  localhost:  │             │  │                 │  │ │
│  │  3000        │             │  │  Preprocessor   │  │ │
│  └──────────────┘             │  │  NER Engine     │  │ │
│                               │  │  Sentiment      │  │ │
│                               │  │  Topic Modeler  │  │ │
│                               │  │  Network        │  │ │
│                               │  └────────┬────────┘  │ │
│                               │           │           │ │
│                               │  ┌────────▼────────┐  │ │
│                               │  │   SQLite DB     │  │ │
│                               │  │  knowledge.db   │  │ │
│                               │  └─────────────────┘  │ │
│                               └───────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │            ML MODELS (loaded into RAM/VRAM)       │  │
│  │                                                   │  │
│  │  Nomio4640/ner-mongolian          (~400MB)        │  │
│  │  cardiffnlp/twitter-xlm-roberta   (~1.1GB)        │  │
│  │  paraphrase-multilingual-mpnet     (~1GB)         │  │
│  │                                                   │  │
│  │  Total: ~2.5GB RAM (or GPU VRAM if available)     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: This is a **two-process architecture**. The frontend and backend are separate programs that talk to each other over HTTP. This is how most modern web applications work.

---

## 3. Layer 1: The Frontend (Next.js + React)

### What is React?

React is a JavaScript library for building user interfaces. The core idea: **your UI is a function of your data**.

Instead of manually updating the HTML when something changes (like jQuery does), you describe what the UI should look like for any given state, and React figures out what changed and updates only those parts.

```
State (data) ──► React ──► UI (what you see)
```

### What is Next.js?

Next.js is a **framework on top of React**. React alone is just a UI library — it doesn't know how to:
- Serve HTML files
- Route between pages (`/` vs `/admin`)
- Optimize for production
- Proxy API requests

Next.js adds all of this. Think of it as: **React = engine, Next.js = the whole car**.

### What is TypeScript?

TypeScript = JavaScript + types. Instead of:
```javascript
// JavaScript — anything goes, errors at runtime
function add(a, b) { return a + b; }
add("hello", 5); // "hello5" — no error until you notice the bug
```

You write:
```typescript
// TypeScript — compiler catches mistakes
function add(a: number, b: number): number { return a + b; }
add("hello", 5); // ❌ ERROR at compile time: "hello" is not a number
```

### How This Frontend Works

The entire main page is **one big React component** in `frontend/src/app/page.tsx` (~1095 lines).

**State Management** — React `useState` hooks at the top track everything:

```typescript
const [result, setResult] = useState<AnalysisResult | null>(null);  // analysis results
const [loading, setLoading] = useState(false);                       // spinner on/off
const [error, setError] = useState("");                              // error message
const [runNer, setRunNer] = useState(true);                          // NER checkbox
const [runSentiment, setRunSentiment] = useState(true);              // Sentiment checkbox
const [runTopics, setRunTopics] = useState(true);                    // Topics checkbox
const [activeTab, setActiveTab] = useState("overview");              // which tab is shown
```

Think of these as **variables that trigger a UI re-render when they change**. When `setLoading(true)` is called, React sees the change and re-renders the spinner.

**The Upload Flow:**

```
User drops CSV file
    ↓
handleUpload() runs
    ↓
setLoading(true)  → spinner appears
    ↓
fetch(`${API_BASE}/api/upload?run_ner=${runNer}&run_sentiment=${runSentiment}&run_topics=${runTopics}`)
    ↓
Wait for response...
    ↓
setResult(data)   → results appear
setLoading(false) → spinner disappears
```

**The Checkboxes:**

```typescript
<input type="checkbox" checked={runSentiment}
       onChange={(e) => setRunSentiment(e.target.checked)} />
```

When you check/uncheck "Сэтгэгдэл (Sentiment)", it sets `runSentiment` to true/false. This value is sent to the backend as a **query parameter**: `?run_sentiment=true`.

**Component Structure:**

```
layout.tsx              ← HTML wrapper + navigation bar
└── page.tsx            ← Main dashboard (the big one)
    ├── Feature Checkboxes (Sentiment, NER, Topics)
    ├── CSV Upload Zone (drag & drop)
    ├── Text Input Area
    ├── Loading Spinner
    ├── Error Banner
    ├── Results Section
    │   ├── Stats Grid (4 cards: total, pos, neu, neg)
    │   ├── Tabs
    │   │   ├── Overview Tab
    │   │   │   ├── Sentiment Pie/Bar Chart (SVG)
    │   │   │   ├── NetworkGraph component (SVG)
    │   │   │   └── Topic Summary
    │   │   ├── Documents Tab (list of all docs with sentiment badges)
    │   │   ├── Insights Tab (auto-generated insights)
    │   │   └── History Tab (past analyses)
    │   └── AnnotationEditor (modal for editing entity/sentiment)
    └── admin/page.tsx  ← Admin panel (separate route)
```

**The NetworkGraph component** is interesting — it draws an SVG graph manually, positioning entities in concentric rings by type (PER inner, ORG next, etc.) and drawing lines between co-occurring entities. No graph library — pure math:

```typescript
const angle = offset + (2 * Math.PI * i) / Math.max(group.length, 1);
posMap.set(node.id, {
  x: cx + r * Math.cos(angle),  // polar → cartesian coordinates
  y: cy + r * Math.sin(angle),
});
```

### What `"use client"` Means

At the top of page.tsx: `"use client"`. This tells Next.js: "This component runs in the browser, not on the server." Without it, Next.js would try to render it on the server side (Server-Side Rendering / SSR), which fails because `useState`, `useEffect`, and `fetch` are browser-only features.

---

## 4. Layer 2: The Network Layer (HTTP, APIs, Proxying)

### What is HTTP?

HTTP (HyperText Transfer Protocol) is how the browser talks to the server. Every time you see a website, your browser is making HTTP requests:

```
Browser: "GET /api/health" (please give me health status)
Server:  "200 OK" + {"status": "ok", "gpu": false}

Browser: "POST /api/upload" + [CSV file bytes]
Server:  "200 OK" + {documents: [...], sentiment_summary: {...}}
```

Key HTTP concepts in this project:
- **GET** = "give me data" (health check, history list)
- **POST** = "here's data, process it" (upload CSV, analyze text)
- **PATCH** = "update this specific thing" (edit a document's annotations)
- **DELETE** = "remove this" (delete analysis history)
- **Status codes**: 200 = success, 400 = your request is wrong, 404 = not found, 500 = server error

### What is an API?

API = Application Programming Interface. It's a **contract** between the frontend and backend:

"If you send me a POST request to `/api/upload` with a CSV file and these query parameters, I'll send you back a JSON object with this exact structure."

The contract is defined by **Pydantic schemas** in the backend (`schemas.py`) and **TypeScript interfaces** in the frontend (`page.tsx`). Both sides must agree on the shape of the data.

### What is REST?

This project follows **REST** (Representational State Transfer) — a convention for organizing API endpoints:

```
GET    /api/health              → check if server is alive
POST   /api/upload              → create a new analysis from CSV
POST   /api/analyze             → create a new analysis from text
GET    /api/history             → list all analyses
GET    /api/history/{id}        → get one specific analysis
DELETE /api/history/{id}        → delete an analysis
PATCH  /api/documents/{doc_id}  → update a document's annotations
```

### The Proxy Problem (Your ECONNRESET Bug)

```
Browser (3000)  →  Next.js Proxy  →  FastAPI (8000)
                   ↑
                   This middleman has a ~30s timeout!
```

The original code set `API_BASE = ""`, which means fetch calls went to `localhost:3000` (the Next.js server). Next.js had a **rewrite rule** that forwarded `/api/*` requests to `localhost:8000` (the backend). This works fine for fast requests but breaks when analysis takes 50+ seconds because the proxy times out.

The fix: `API_BASE = "http://localhost:8000"` — the browser talks to the backend **directly**, no middleman.

### What is CORS?

Cross-Origin Resource Sharing. When your browser at `localhost:3000` tries to fetch from `localhost:8000`, the browser blocks it by default (security measure). The backend must explicitly say "I allow requests from other origins":

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow any origin (permissive for development)
)
```

---

## 5. Layer 3: The Backend API (FastAPI + Uvicorn)

### What is FastAPI?

FastAPI is a Python web framework. It lets you write functions that respond to HTTP requests:

```python
@app.get("/api/health")          # When someone sends GET to /api/health
async def health():              # Run this function
    return {"status": "ok"}      # And send back this JSON
```

**Why FastAPI specifically?**
1. **Type validation** — Pydantic models automatically validate incoming data
2. **Async support** — `async def` means it can handle multiple requests simultaneously
3. **Auto-documentation** — visit `localhost:8000/docs` and you get interactive API docs free
4. **Fast** — one of the fastest Python web frameworks

### What is Uvicorn?

FastAPI is just code — it doesn't know how to listen on a network port. Uvicorn is the **ASGI server** that:
1. Opens port 8000
2. Listens for incoming HTTP connections
3. Translates HTTP requests into Python function calls
4. Sends the function's return value back as an HTTP response

```
Network packet → Uvicorn → FastAPI → Your function → FastAPI → Uvicorn → Network packet
```

The command `uvicorn main:app --reload` means: "Start Uvicorn, find the `app` object in `main.py`, and restart when code changes."

### What is `async`?

When you write `async def upload_csv(...)`, Python doesn't block the entire server while this function runs. If function A is waiting for a database query, Python can run function B in the meantime. This is **concurrency** (not parallelism — it's one thread doing multiple things by switching between them when one is waiting).

However, your NLP models are CPU-bound (they actually compute things, not just wait). That's why the `_run_analysis()` function is a regular `def` (not async) — FastAPI automatically runs it in a thread pool.

### The Singleton Pattern in services.py

```python
ner       = NEREngine()      # Created ONCE when the module loads
sentiment = SentimentAnalyzer()
topic     = TopicModeler()
```

These are **module-level singletons**. In Python, when you `import services`, the code at module level runs exactly once. After that, every import reuses the same objects. This is critical because:

1. Each model is ~400MB-1GB in RAM
2. Loading takes 10-30 seconds
3. You only want to load once, not per request

### Routers: Organizing Endpoints

FastAPI uses **routers** to group related endpoints:

```python
# main.py
app.include_router(analysis.router, prefix="/api")      # /api/upload, /api/analyze, etc.
app.include_router(insights.router, prefix="/api")       # /api/insights
app.include_router(admin.router,    prefix="/api/admin") # /api/admin/knowledge, etc.
```

This is like folders for your endpoints. Instead of one giant file, related endpoints live together.

### Pydantic: Data Validation

```python
class TextAnalysisRequest(BaseModel):
    text: str                     # MUST be a string
    run_ner: bool = True          # Boolean, defaults to True if not provided
    run_sentiment: bool = True
    run_topics: bool = False
```

When someone sends a POST to `/api/analyze`, Pydantic automatically:
1. Parses the JSON body
2. Checks that `text` is a string (rejects if missing)
3. Fills in defaults for `run_ner` etc. if not provided
4. Returns a 422 error with details if validation fails

This eliminates an entire class of bugs (missing fields, wrong types).

---

## 6. Layer 4: Text Preprocessing (The Unsung Hero)

This is the most **underappreciated** layer. Bad preprocessing = bad results, no matter how good your models are.

### Why Preprocessing Matters

Raw social media text is messy:

```
Input:  "б.амар энэ яаж бхаа вэ 😡😡 https://t.co/xyz #монгол @user123"
```

A BERT model can't read emojis, URLs are noise, and lowercase `б.амар` won't be recognized as a person name because BERT expects capitalization. Preprocessing transforms this into clean text that models can understand.

### The Dual Pipeline — Why Two Versions?

Each text is preprocessed **twice** because different NLP tasks need different things:

```
Raw text: "б.амар энэ яаж бхаа вэ 😡😡 https://t.co/xyz"
                    │
           ┌───────┴───────┐
           ▼               ▼
    NLP Version        TM Version
    (for NER &        (for Topic
     Sentiment)        Modeling)
           │               │
           ▼               ▼
"Б.Амар энэ яаж    "амар яаж"
 бхаа вэ [ANGRY]
 [URL]"
```

**NLP version** (for NER + Sentiment):
- Keeps punctuation (helps BERT understand sentence structure)
- Capitalizes names (`б.амар` → `Б.Амар`) — critical for NER
- Converts emojis to text markers (`😡` → `[ANGRY]`) — preserves sentiment signal
- Keeps stopwords ("бхаа вэ") — grammar words help sentiment
- Keeps URLs as `[URL]` token

**TM version** (for Topic Modeling):
- Removes all punctuation
- Lowercases everything
- Removes stopwords (grammar words dilute topics)
- Removes URLs, emojis entirely
- Strips name initials (`б_амар` → `амар`)

### The Name Protection System

Mongolian names have a unique format: `А.Бат-Эрдэнэ` (Initial.Surname). The dot and hyphen would be destroyed by punctuation cleaning. The solution:

```
Step 1 (protect):   А.Бат-Эрдэнэ → А_Бат-Эрдэнэ   (underscore survives cleaning)
Step 2 (clean):     remove other punctuation
Step 3 (restore):   А_Бат-Эрдэнэ → А.Бат-Эрдэнэ   (NLP mode only)
```

For lowercase social media names:
```
б.амар → Б_Амар (capitalize + protect) → Б.Амар (restore)
```

This is done with **regex** (regular expressions):

```python
# Match uppercase initial: А.Бат, А.Бат-Эрдэнэ
MN_NAME_UPPER = re.compile(r"\b([А-ЯӨҮЁ])\.\s*([А-Яа-яӨөҮүЁё][а-яөүёa-z]+(?:-[А-Яа-яӨөҮүЁё][а-яөүёa-z]+)*)")

# Match lowercase initial: б.амар (very common in social media)
MN_NAME_LOWER = re.compile(r"\b([а-яөүё])\.\s*([а-яөүёa-z]+(?:-[а-яөүёa-z]+)*)")
```

### Stopwords

Stopwords are words so common they carry no meaning:
- English: "the", "is", "at", "which"
- Mongolian: "байна" (is), "энэ" (this), "гэж" (that), "юм" (is)

For topic modeling, you MUST remove these or every topic will just be "байна гэж юм". The project has 80+ hardcoded Mongolian stopwords, plus the admin can add more via the UI.

---

## 7. Layer 5: NLP Models — The Brain

### What is a Neural Network? (Simple Version)

A neural network is a mathematical function that learns patterns from examples:

```
Training:
  Input: "Энэ кино үнэхээр гайхалтай"  →  Label: positive
  Input: "Маш муу үйлчилгээ"            →  Label: negative
  ... thousands more examples ...

The network adjusts its internal numbers (parameters/weights) until
it can correctly predict the label from the input.

Using:
  Input: "Сайхан байна шүү" → Network → "positive" (87% confident)
```

The "learning" is math: each example slightly adjusts millions of numbers (weights) to reduce the error between predicted and actual labels. After millions of examples, the network captures patterns of language.

### What is BERT?

BERT (Bidirectional Encoder Representations from Transformers) is a specific neural network architecture. Think of it as a very sophisticated pattern recognizer that understands language:

```
"Гэрээг         Батболд          байгуулсан"
 (contract)     (Batbold)        (established)
     ↓              ↓                 ↓
   ┌────────────────────────────────────┐
   │              BERT                   │
   │                                     │
   │  Reads ALL words simultaneously     │
   │  (not left-to-right like a human)   │
   │                                     │
   │  Each word's meaning is informed    │
   │  by ALL other words in the sentence │
   └────────────────────────────────────┘
     ↓              ↓                 ↓
   O (other)     B-PER (person)    O (other)
```

**Key BERT concepts:**
- **Tokens**: BERT splits text into subwords. "Батболд" might become ["Бат", "##болд"]. The `##` means "continuation of previous word."
- **512 token limit**: BERT can only process 512 tokens at once. Longer text is truncated (entities in the second half are lost).
- **Fine-tuning**: BERT is pre-trained on massive text data, then fine-tuned on specific tasks (NER, sentiment).

### What is a Transformer?

The core mechanism inside BERT. The key innovation is **attention**: the model learns which other words to "pay attention to" when processing each word.

```
"Батболд Монгол улсын ерөнхийлөгч"

When processing "ерөнхийлөгч" (president):
  - High attention to "Батболд" (who is the president?)
  - High attention to "Монгол улсын" (which country?)
  - Low attention to function words
```

This is what makes BERT so powerful — it understands **context**, not just individual words.

### NER Engine — Finding Named Entities

**Model**: `Nomio4640/ner-mongolian` — a BERT model fine-tuned on Mongolian NER data.

**What it does**: Takes a sentence, outputs entity labels for each token:

```
Input:  "Б.Амар Улаанбаатарт ирсэн"
Output: [
  {word: "Б.Амар",        entity_group: "PER", score: 0.95},
  {word: "Улаанбаатарт",  entity_group: "LOC", score: 0.89},
]
```

Entity types:
- **PER** — Person names
- **ORG** — Organizations
- **LOC** — Locations
- **MISC** — Other named entities

**The Long Text Problem** (solved in ner_engine.py):

BERT's 512-token limit means long posts get silently truncated. A 2000-character Facebook post would lose everything after ~1300 characters. The solution:

```
Long text (2500 chars)
    ↓
Split at sentence boundaries into chunks of ≤1300 chars
    ↓
Chunk 1 (chars 0-1250)     → NER → entities with offsets 0-1250
Chunk 2 (chars 1251-2500)  → NER → entities with offsets 0-1249
    ↓
Fix offsets: Chunk 2 entities get +1251 added to their positions
    ↓
Deduplicate entities at chunk boundaries
    ↓
Merged entity list with correct positions in original text
```

**Batch Processing** for GPU efficiency:

```python
def recognize_batch(self, texts, batch_size=16):
    # Short texts: batch together for GPU parallelism
    # Long texts: chunk individually
```

GPUs are fast at processing many inputs simultaneously. Sending 16 short texts at once is much faster than sending them one by one. But long texts must be handled individually because of chunking.

### Sentiment Analyzer — Reading Emotions

**Model**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

This is an **XLM-RoBERTa** model — a multilingual version of RoBERTa (which is an improved BERT). "XLM" means it was trained on text from ~100 languages simultaneously.

**Why this specific model?**
- Trained on Twitter data → understands informal social media language
- Multilingual → works on Mongolian even though it wasn't specifically trained on it
- This is called **cross-lingual transfer**: patterns learned from English/French/etc. generalize to Mongolian

```python
# The model outputs raw labels like "LABEL_0", "LABEL_1", "LABEL_2"
# These are mapped to human-readable labels:
LABEL_MAP = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}
```

**Truncation**: `max_length=512` — if a text is longer than 512 tokens, it's truncated (only the first 512 tokens are analyzed). Unlike NER, sentiment analysis doesn't need exact character positions, so this is acceptable — the beginning of a text usually captures the overall sentiment.

### Topic Modeler — Finding What People Talk About

**Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` + BERTopic

Topic modeling is more complex than NER/sentiment because it involves multiple steps:

```
Texts → [Embedding] → [Clustering] → [Keyword Extraction] → Topics

Step 1: Embedding
  "Газрын тосны үнэ өссөн"  → [0.23, -0.45, 0.78, ..., 0.12]  (768 numbers)
  "Нефтийн зардал нэмэгдсэн" → [0.21, -0.43, 0.76, ..., 0.14]  (768 numbers)

  Similar meanings → similar numbers (these two are about the same thing)

Step 2: Clustering (KMeans or HDBSCAN)
  Group the 768-dimensional points into clusters.
  Documents that land in the same cluster = same topic.

Step 3: Keyword Extraction (c-TF-IDF)
  For each cluster, find which words are distinctive.
  Cluster A: "газрын тосны", "нефть", "үнэ" → Topic: Oil Prices
  Cluster B: "сонгууль", "нам", "дуудлага"   → Topic: Elections
```

**What is an Embedding?**

An embedding converts text into a list of numbers (a vector) where **meaning is encoded as position in space**. Texts with similar meanings end up close together:

```
                    ↑
  "нохой"  ●       | "муур"  ●     ← Animals cluster together
                    |
  "машин" ●        | "онгоц" ●     ← Vehicles cluster together
                    |
  ──────────────────┼──────────────►
```

The sentence-transformer model produces 768-dimensional embeddings (768 numbers per text). In 768 dimensions, there's enough room to encode very nuanced meaning differences.

**What is BERTopic?**

BERTopic is a framework that combines:
1. **Sentence embeddings** (from sentence-transformers)
2. **Dimensionality reduction** (UMAP — compresses 768D → 5D for clustering)
3. **Clustering** (HDBSCAN or KMeans — finds groups)
4. **c-TF-IDF** (extracts keywords that define each topic)
5. **MMR** (Maximal Marginal Relevance — picks diverse keywords, not synonyms)

**The Small Dataset Problem:**

HDBSCAN (the default clustering algorithm) needs a minimum cluster size. With <50 documents, it assigns everything to "outlier topic -1" (uncategorized). The solution:

```python
if n_docs >= 50:
    # HDBSCAN — good for large datasets, finds natural cluster shapes
    cluster_model = HDBSCAN(...)
else:
    # KMeans — forces every document into a cluster (no outliers)
    n_clusters = max(2, min(n_docs // 3, 10))
    cluster_model = KMeans(n_clusters=n_clusters)
```

**Mongolian Suffix Stripping:**

Mongolian is an **agglutinative language** — suffixes are attached to words to indicate grammar:
- монгол → монголын (genitive), монголд (dative), монголоос (ablative)

Without stripping, BERTopic sees these as three different words. The custom tokenizer strips common suffixes:

```python
_MN_SUFFIXES = ["аас", "ээс", "оос", "өөс", "ийн", "ын", "ний", ...]

def _mn_stem(word):
    for sfx in _MN_SUFFIXES:
        if word.endswith(sfx) and len(word) - len(sfx) >= 3:
            return word[:-len(sfx)]   # "монголын" → "монгол"
    return word
```

### Network Analyzer — Mapping Relationships

This is the simplest component — pure Python, no ML models:

```python
# For each document, look at which entities appear together:
doc1: [Батболд (PER), Монгол банк (ORG)]
doc2: [Батболд (PER), Улаанбаатар (LOC)]
doc3: [Батболд (PER), Монгол банк (ORG), Улаанбаатар (LOC)]

# Count co-occurrences:
(Батболд, Монгол банк):  2 times
(Батболд, Улаанбаатар):  2 times
(Монгол банк, Улаанбаатар): 1 time

# Build graph:
Nodes: Батболд (freq: 3), Монгол банк (freq: 2), Улаанбаатар (freq: 2)
Edges: Батболд──Монгол банк (weight: 2), Батболд──Улаанбаатар (weight: 2), ...
```

The algorithm uses `itertools.combinations` to generate all pairs of entities within each document, then counts how many documents each pair co-appears in.

---

## 8. Layer 6: Data Persistence (SQLite)

### What is SQLite?

SQLite is a **file-based database**. Unlike PostgreSQL or MySQL (which run as separate servers), SQLite stores everything in a single file (`knowledge.db`). It's embedded in Python's standard library — no installation needed.

**Why SQLite for this project?**
- Zero configuration (just create a file)
- Perfect for single-user applications
- Fast enough for this scale (thousands of documents)
- The entire database is one file — easy to backup, share, or delete

### The Tables

```sql
-- Who said what and how they felt
analysis_sessions     → One row per upload (when, from what file, summary)
analysis_documents    → One row per text in that upload (text, sentiment, entities, topics)

-- Admin configuration
stopwords            → Words to filter out during topic modeling
custom_labels        → Rename entity types (PER → "Улс төрч")
knowledge_entries    → General knowledge base entries
```

### WAL Mode

```python
conn.execute("PRAGMA journal_mode=WAL")
```

WAL = Write-Ahead Logging. SQLite's default mode locks the entire database during writes (no one can read while someone is writing). WAL mode allows **concurrent readers** — important because FastAPI handles multiple requests simultaneously.

### The Relationship Between Tables

```
analysis_sessions (1) ──────────► (many) analysis_documents
     session_id           FOREIGN KEY + ON DELETE CASCADE
```

`ON DELETE CASCADE` means: when you delete a session, all its documents are automatically deleted too. No orphan rows.

---

## 9. How a Single Request Flows Through Everything

Let's trace exactly what happens when you upload a 500-row CSV with only Sentiment checked:

```
1. BROWSER
   User drops "comments.csv" on the upload zone
   JavaScript reads the file into memory
   fetch("http://localhost:8000/api/upload?run_ner=false&run_sentiment=true&run_topics=false",
         {method: "POST", body: formData})

2. NETWORK
   Browser creates HTTP POST request
   TCP connection to localhost:8000
   Request travels through OS networking stack (loopback interface)

3. UVICORN
   Receives raw bytes on port 8000
   Parses HTTP: method=POST, path=/api/upload, query params, body
   Matches route to upload_csv() function

4. FASTAPI (analysis.py → upload_csv)
   Validates: file ends with .csv? ✓
   Reads file bytes: await file.read()
   Parses CSV: csv.DictReader(...)
   Finds text column: "text" or "Text"
   Calls _run_analysis(rows, "Text", run_ner=False, run_sentiment=True, run_topics=False)

5. ANALYSIS PIPELINE (_run_analysis)

   5a. PREPROCESSING (317ms for 500 rows)
       For each of 500 texts:
         preprocessor.preprocess_dual(raw_text) → (nlp_text, tm_text)
         - normalize unicode
         - protect names (А.Бат → А_Бат)
         - remove URLs, hashtags, emoji
         - NLP: capitalize, restore names
         - TM: deep clean, lowercase, remove stopwords
       Result: 500 nlp_texts + 500 tm_texts

   5b. NER — SKIPPED (run_ner=False)

   5c. SENTIMENT (50,136ms for 500 rows)
       services.sentiment.analyze_batch(nlp_texts)
       - Pipeline already loaded (warmup ran at startup)
       - Tokenize all 500 texts → subword tokens
       - Feed through XLM-RoBERTa in batches of 16
       - Each batch: tokens → model → probability distribution [neg, neu, pos]
       - Pick highest probability as label
       Result: 500 SentimentResults (175 pos, 183 neu, 142 neg)

   5d. TOPICS — SKIPPED (run_topics=False)

   5e. NETWORK — SKIPPED (depends on NER which was skipped)

   5f. ASSEMBLE RESPONSE
       Combine all results into AnalysisResponse object
       Calculate sentiment_summary: {positive: 175, neutral: 183, negative: 142}

6. DB PERSISTENCE (_save_and_attach_doc_ids)
   INSERT INTO analysis_sessions (source_filename, total_documents, ...)
   INSERT INTO analysis_documents (session_id, raw_text, sentiment_label, ...) × 500
   Attach doc_ids back to response

7. FASTAPI RESPONSE
   AnalysisResponse → JSON serialization (Pydantic handles this)
   HTTP 200 response with JSON body

8. BROWSER
   fetch() promise resolves
   setResult(data) → React re-renders with results
   setLoading(false) → spinner disappears
   Stats cards show: 500 total, 175 positive, 183 neutral, 142 negative
```

**Time breakdown:**
- Preprocessing: 317ms (0.6ms per text)
- Sentiment: 50,136ms (100ms per text)
- DB save: ~100ms
- **Total: ~50.5 seconds**

The models are the bottleneck — 99.3% of time is spent on sentiment inference.

---

## 10. The ECONNRESET Bug — What Actually Happened

```
Timeline:
0s     Browser sends request to localhost:3000 (Next.js)
0s     Next.js proxy forwards to localhost:8000 (FastAPI)
0.3s   Preprocessing complete
30s    ⚠️ Next.js proxy timeout! Kills the connection to browser
       Browser receives: "Error: socket hang up" (ECONNRESET)
50s    FastAPI finishes sentiment analysis, sends 200 OK
       ... but Next.js already closed the connection, nobody is listening
       FastAPI logs "200 OK" because IT succeeded
```

**ECONNRESET** = "connection reset" — one side forcibly closed the TCP connection. The Next.js proxy closed the connection to the browser after ~30s, and when FastAPI tried to send the response back through the proxy, the proxy had already torn down that connection.

**Why the backend logs "200 OK"**: The backend processed everything correctly. It doesn't know or care that the proxy between it and the browser timed out. From FastAPI's perspective, the analysis succeeded and the response was sent.

**The fix**: Skip the proxy entirely. `API_BASE = "http://localhost:8000"` makes the browser talk directly to FastAPI, where `fetch()` has no built-in timeout.

---

## 11. Architecture Diagram

### Technical Stack Diagram

```
┌─────────────── PRESENTATION LAYER ───────────────┐
│                                                    │
│   Next.js 16 + React 19 + TypeScript 5            │
│   ┌──────────────────────────────────────────┐    │
│   │  page.tsx (Dashboard)                     │    │
│   │  - useState for state management          │    │
│   │  - fetch() for API calls                  │    │
│   │  - SVG for charts & network graph         │    │
│   ├──────────────────────────────────────────┤    │
│   │  AnnotationEditor.tsx (Entity editing)    │    │
│   ├──────────────────────────────────────────┤    │
│   │  admin/page.tsx (Admin panel)             │    │
│   └──────────────────────────────────────────┘    │
│   globals.css (Dark theme, CSS variables)          │
│                                                    │
└──────────────────────┬─────────────────────────────┘
                       │ HTTP (JSON over REST)
                       │ port 8000
┌──────────────────────▼─────────────────────────────┐
│              API LAYER (Adapters)                    │
│                                                      │
│   FastAPI + Uvicorn (ASGI server)                   │
│   ┌────────────────────────────────────────────┐    │
│   │  main.py                                    │    │
│   │  - CORS middleware                          │    │
│   │  - Exception handler                        │    │
│   │  - Model warmup on startup                  │    │
│   ├────────────────────────────────────────────┤    │
│   │  routers/                                   │    │
│   │    analysis.py  → /upload, /analyze, /history│   │
│   │    insights.py  → /insights                 │    │
│   │    admin.py     → /admin/*                  │    │
│   ├────────────────────────────────────────────┤    │
│   │  schemas.py  (Pydantic request/response)    │    │
│   │  services.py (Singleton ML model instances) │    │
│   └────────────────────────────────────────────┘    │
│                                                      │
└──────────────────────┬───────────────────────────────┘
                       │ Python function calls
┌──────────────────────▼───────────────────────────────┐
│              NLP CORE (Domain Logic)                  │
│                                                       │
│   models.py  — Plain dataclasses (no framework deps) │
│   ┌─────────────────────────────────────────────┐    │
│   │  preprocessing.py                            │    │
│   │  - Mongolian text cleaning                   │    │
│   │  - Name protection/restoration               │    │
│   │  - Dual pipeline (NLP vs TM)                 │    │
│   │  - Stopword removal                          │    │
│   │  - Emoji → sentiment markers                 │    │
│   ├─────────────────────────────────────────────┤    │
│   │  ner_engine.py                               │    │
│   │  - HuggingFace Transformers pipeline         │    │
│   │  - Nomio4640/ner-mongolian (BERT)            │    │
│   │  - Long text chunking (>1300 chars)          │    │
│   │  - Batch processing                          │    │
│   ├─────────────────────────────────────────────┤    │
│   │  sentiment.py                                │    │
│   │  - XLM-RoBERTa (twitter-sentiment)           │    │
│   │  - LABEL_0/1/2 → negative/neutral/positive   │    │
│   │  - Batch processing                          │    │
│   ├─────────────────────────────────────────────┤    │
│   │  topic_modeler.py                            │    │
│   │  - BERTopic + sentence-transformers          │    │
│   │  - KMeans fallback for <50 docs              │    │
│   │  - Mongolian suffix stripping                │    │
│   │  - c-TF-IDF keyword extraction               │    │
│   ├─────────────────────────────────────────────┤    │
│   │  network_analyzer.py                         │    │
│   │  - Entity co-occurrence counting             │    │
│   │  - Graph construction (NetworkX logic)       │    │
│   ├─────────────────────────────────────────────┤    │
│   │  knowledge_base.py                           │    │
│   │  - SQLite operations                         │    │
│   │  - Analysis persistence                      │    │
│   │  - Admin CRUD                                │    │
│   └─────────────────────────────────────────────┘    │
│                                                       │
└───────────────────────┬───────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────┐
│              STORAGE LAYER                             │
│                                                        │
│   knowledge.db (SQLite, WAL mode)                     │
│   ┌──────────────────────────────────────────────┐    │
│   │  analysis_sessions     (history)              │    │
│   │  analysis_documents    (per-doc results)      │    │
│   │  stopwords             (configurable)         │    │
│   │  custom_labels         (PER → "Улс төрч")    │    │
│   │  knowledge_entries     (admin knowledge tree) │    │
│   └──────────────────────────────────────────────┘    │
│                                                        │
│   HuggingFace Model Cache (~/.cache/huggingface/)     │
│   ┌──────────────────────────────────────────────┐    │
│   │  ner-mongolian          (~400MB)              │    │
│   │  twitter-xlm-roberta    (~1.1GB)              │    │
│   │  multilingual-mpnet     (~1GB)                │    │
│   └──────────────────────────────────────────────┘    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Logical Data Flow Diagram

```
          USER
           │
      ┌────▼────┐
      │  Upload  │──────────────────────────────┐
      │  CSV or  │                              │
      │  Text    │                              │
      └────┬─────┘                              │
           │                                    │
      ┌────▼──────────────┐                     │
      │  DUAL PREPROCESS  │                     │
      │  ┌──────┬───────┐ │                     │
      │  │ NLP  │  TM   │ │                     │
      │  │ text │ text  │ │                     │
      │  └──┬───┴───┬───┘ │                     │
      └─────┼───────┼─────┘                     │
            │       │                           │
     ┌──────┼───────┼──────────────────────┐    │
     │      │       │     ANALYSIS         │    │
     │  ┌───▼───┐ ┌─▼──────────┐          │    │
     │  │  NER  │ │   Topic    │          │    │
     │  │ BERT  │ │  BERTopic  │          │    │
     │  └───┬───┘ └─────┬──────┘          │    │
     │      │           │                  │    │
     │  ┌───▼────────┐  │                  │    │
     │  │  Sentiment │  │                  │    │
     │  │  RoBERTa   │  │                  │    │
     │  └───┬────────┘  │                  │    │
     │      │           │                  │    │
     │  ┌───▼───┐       │                  │    │
     │  │Network│       │                  │    │
     │  │ Graph │       │                  │    │
     │  └───┬───┘       │                  │    │
     └──────┼───────────┼──────────────────┘    │
            │           │                       │
      ┌─────▼───────────▼───────┐               │
      │   COMBINE & SAVE TO DB  │◄──────────────┘
      └─────────┬───────────────┘    (feature flags decide
                │                     which models run)
      ┌─────────▼───────────────┐
      │    RETURN JSON TO UI    │
      │                         │
      │  documents: [           │
      │    {text, entities,     │
      │     sentiment, topic}   │
      │  ],                     │
      │  network: {nodes,edges} │
      │  topic_summary: [...]   │
      │  sentiment_summary: {}  │
      └─────────────────────────┘
```

---

## 12. What You Don't Know Yet (Knowledge Gaps)

Based on reading your project, here are the areas where you'd benefit from deeper understanding, roughly prioritized by impact:

### Critical Gaps (Must Learn)

1. **Git & Version Control** — Your project has submodules and a minimal commit history. Learn branching, meaningful commits, `.gitignore`, and how to collaborate with others.

2. **Environment Management** — `requirements.txt`, virtual environments (`venv`/`conda`), pinning dependency versions. Your project would break if someone has a different version of `transformers`.

3. **Error Handling Patterns** — The project has many bare `except Exception` blocks that silently swallow errors. Learn when to catch, what to catch, and how to make errors visible.

4. **Async Programming** — You're using FastAPI (async) but `_run_analysis` is synchronous. Learn what `async/await` actually means, why it matters, and when to use `run_in_executor`.

5. **Testing** — Zero tests in this project. Learn pytest, how to test API endpoints, how to test NLP pipelines without running the full model.

### Important Gaps (Should Learn)

6. **Deployment** — How to run this on a real server (Docker, cloud VMs, nginx). Right now it only runs on localhost.

7. **Security** — CORS is `*`, no authentication, SQL injection possible in admin endpoints. Learn OWASP basics.

8. **Performance Profiling** — Why does sentiment take 100ms per text? Is it CPU-bound? Memory-bound? Batching helps but you need to understand WHY.

9. **React Patterns** — The 1095-line page.tsx should be broken into smaller components. Learn component composition, custom hooks, state management patterns.

10. **Database Design** — SQLite is fine for now, but learn about indexes, query optimization, and when you'd need to move to PostgreSQL.

### Nice to Have

11. **ML Operations (MLOps)** — Model versioning, A/B testing models, monitoring model performance in production.

12. **Data Pipeline Design** — How to handle streaming data, scheduled batch processing, data quality checks.

13. **CSS Architecture** — The styles work but are fragile. Learn CSS modules, CSS-in-JS, or Tailwind.

---

## 13. Improvement Plan

### As an NLP Expert

| Priority | Improvement | Why | Effort |
|----------|-------------|-----|--------|
| 🔴 HIGH | **Fine-tune sentiment on Mongolian data** | The current model is multilingual but not Mongolian-specific. Accuracy would jump 10-20% with 5K labeled Mongolian sentiment examples. | 2-3 weeks |
| 🔴 HIGH | **Train a better Mongolian NER model** | `Nomio4640/ner-mongolian` is fine but limited. Collect Mongolian NER training data from news/social media. | 1-2 months |
| 🟡 MED | **Add aspect-based sentiment** | "Хоол сайн, үйлчилгээ муу" has BOTH positive and negative. Current model gives one label. Aspect-based sentiment would say: food=positive, service=negative. | 1 month |
| 🟡 MED | **Add text summarization** | After topic modeling, generate a 2-3 sentence summary of what each topic cluster is about. Use an LLM (GPT-4 / Claude API) for this. | 1-2 weeks |
| 🟡 MED | **Add Mongolian word segmentation** | Mongolian doesn't always have clear word boundaries. A proper tokenizer (like for Chinese/Japanese) would improve all models. | 2-4 weeks |
| 🟢 LOW | **Multi-language support** | Add English/Russian detection and route to appropriate models. | 1 week |
| 🟢 LOW | **Add keyword extraction** | RAKE or KeyBERT to extract key phrases per document, not just per topic. | 3 days |

### As a Web Developer Expert

| Priority | Improvement | Why | Effort |
|----------|-------------|-----|--------|
| 🔴 HIGH | **Add background job processing** | Analysis blocks the API thread. Use Celery + Redis for async jobs: upload → return job ID → poll for status. | 1-2 weeks |
| 🔴 HIGH | **Add authentication** | Anyone can access the API. Add JWT tokens, user roles (admin vs analyst). | 1 week |
| 🔴 HIGH | **Split page.tsx into components** | 1095 lines is unmaintainable. Extract: `<UploadZone>`, `<SentimentChart>`, `<DocumentList>`, `<NetworkGraph>`. | 3-5 days |
| 🟡 MED | **Add WebSocket progress** | During 50s analysis, show real-time progress ("Processing document 234/500..."). | 3-5 days |
| 🟡 MED | **Dockerize everything** | `docker-compose up` should start both frontend and backend. Essential for deployment. | 2-3 days |
| 🟡 MED | **Add proper error boundaries** | Frontend crashes on any unhandled error. React Error Boundaries + toast notifications. | 2 days |
| 🟡 MED | **Replace SVG network with a graph library** | D3.js or vis.js would give you zoom, drag, physics simulation for free. | 3-5 days |
| 🟢 LOW | **Add data export** | Download results as CSV/Excel/PDF. | 2-3 days |
| 🟢 LOW | **Add comparison view** | Compare sentiment between two CSV uploads over time. | 1 week |

### As a Hardware Expert

| Priority | Improvement | Why | Effort |
|----------|-------------|-----|--------|
| 🔴 HIGH | **GPU acceleration** | Your sentiment takes 100ms/text on CPU. On a $200 GPU (RTX 3060), it would be ~5ms/text. 20x speedup. | $200 + 1 day setup |
| 🔴 HIGH | **Model quantization** | Convert FP32 models to INT8 using ONNX Runtime or bitsandbytes. Same accuracy, 2-4x faster, half the RAM. | 2-3 days |
| 🟡 MED | **Batch size tuning** | Current batch_size=16 is arbitrary. Profile with batch sizes 8/16/32/64 to find the sweet spot for your hardware. | 1 day |
| 🟡 MED | **Model caching on SSD** | First model load downloads from HuggingFace. Pre-download to local SSD and point to local path (already partially done for NER). | 1 day |
| 🟢 LOW | **Use TensorRT/ONNX** | Compile models to optimized inference engines. 3-5x faster than PyTorch inference. | 3-5 days |

### As a Business Expert

| Priority | Improvement | Why | Effort |
|----------|-------------|-----|--------|
| 🔴 HIGH | **Define target users** | Government PR teams? News agencies? Marketing firms? Each needs different features. | 1 week research |
| 🔴 HIGH | **Add automated data collection** | Users shouldn't manually export CSVs. Build scrapers for Facebook pages, news sites, Twitter. (With proper permissions.) | 2-4 weeks |
| 🔴 HIGH | **Add scheduled monitoring** | "Check this Facebook page every hour and alert me if negative sentiment spikes." This is the real value proposition. | 2-3 weeks |
| 🟡 MED | **Add report generation** | Auto-generate weekly PDF reports: "Sentiment trends, top entities, emerging topics." | 1-2 weeks |
| 🟡 MED | **Multi-tenant** | Support multiple organizations, each with their own data, stopwords, labels. | 2-3 weeks |
| 🟡 MED | **Add dashboard analytics** | Time-series charts, sentiment trends over days/weeks, entity frequency changes. | 1-2 weeks |
| 🟢 LOW | **API-first design** | Other tools should be able to call your API. Add rate limiting, API keys, documentation. | 1 week |

### What You Should Focus On (Priority Order)

1. **Get GPU working** — Biggest bang for buck. Transforms 50s analysis into 3s.
2. **Add background job processing** — Users shouldn't stare at a spinner for 50s.
3. **Collect Mongolian training data** — The models are the product. Better models = better product.
4. **Add authentication + deployment** — Can't show this to real users without it.
5. **Add automated data collection** — Manual CSV uploads won't scale.
6. **Add scheduled monitoring + alerts** — This is what turns a tool into a product.

---

## 14. Study Resources

### Fundamentals You Need

**Python**
- "Automate the Boring Stuff with Python" (free online) — good for filling Python gaps
- Real Python (realpython.com) — best tutorials for intermediate Python
- "Fluent Python" by Luciano Ramalho — when you're ready for advanced Python

**JavaScript / React / Next.js**
- javascript.info — the best JavaScript tutorial (free)
- React Official Tutorial (react.dev/learn) — new React docs are excellent
- Next.js Tutorial (nextjs.org/learn) — hands-on, builds a real app
- TypeScript Handbook (typescriptlang.org/docs/handbook)

**Web Development Concepts**
- MDN Web Docs (developer.mozilla.org) — the reference for everything web
- "HTTP: The Definitive Guide" — understand the protocol you use every day
- web.dev — Google's web development learning platform

### NLP / ML

**Machine Learning Foundations**
- 3Blue1Brown "Neural Networks" YouTube series — visual intuition
- Andrew Ng's Machine Learning Specialization (Coursera) — the classic
- fast.ai "Practical Deep Learning" (course.fast.ai) — top-down, code-first approach

**NLP Specific**
- HuggingFace NLP Course (huggingface.co/learn/nlp-course) — FREE, covers exactly the tech you use (Transformers, pipelines, fine-tuning)
- "Speech and Language Processing" by Jurafsky & Martin (free draft online) — the NLP textbook
- Jay Alammar's blog (jalammar.github.io) — the best visual explanations of Transformers and BERT

**BERTopic Specific**
- BERTopic documentation (maartengr.github.io/BERTopic) — very well-written
- Maarten Grootendorst's YouTube tutorials on BERTopic

### Software Engineering

**Architecture & Design**
- "Clean Architecture" by Robert Martin — how to structure code into layers
- "Designing Data-Intensive Applications" by Martin Kleppmann — how real systems work at scale

**DevOps / Deployment**
- Docker Getting Started (docs.docker.com/get-started)
- "The Missing Semester of CS Education" (MIT, free) — shell, git, debugging, profiling

**API Design**
- "RESTful Web APIs" by Richardson — good REST design principles
- FastAPI documentation (fastapi.tiangolo.com) — one of the best docs of any framework

### Mongolian NLP Specific

- Look for papers on Mongolian NLP at ACL Anthology (aclanthology.org) — search "Mongolian"
- Universal Dependencies project — has Mongolian treebank data
- MongolNLP GitHub organization — community tools for Mongolian

---

> **Reading time estimate**: This document is ~4500 words of explanation + code + diagrams. At a careful study pace with experimenting in the codebase alongside, plan for 2-3 hours.
>
> **Best way to use this**: Read a section, then open the corresponding source file and trace through the code yourself. Understanding comes from matching the explanation to the actual code.

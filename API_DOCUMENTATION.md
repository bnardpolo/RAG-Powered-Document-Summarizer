# Papers DB - Comprehensive API Documentation

## Overview

The Papers DB is a comprehensive document management and analysis system built with Streamlit. It provides capabilities for document ingestion, text extraction, semantic search, evaluation metrics, and experiment tracking. The system is designed to handle academic papers, notes, and multimedia content with advanced NLP capabilities.

## Architecture

The system consists of 7 core modules:

- **`storage_s3.py`** - AWS S3 integration for file storage and retrieval
- **`stream_app.py`** - Main Streamlit application with UI components and workflows
- **`stream_metrics.py`** - Evaluation metrics for summarization and retrieval tasks
- **`stream_utils_db.py`** - Database operations, schema management, and data persistence
- **`stream_utils_text.py`** - Text extraction, processing, and summarization utilities
- **`stream_voice.py`** - Audio processing and speech-to-text transcription
- **`tracking_mlflow.py`** - MLflow integration for experiment tracking and model management

## Quick Start

```python
# Initialize the database
from stream_utils_db import init_db
conn = init_db("papers.db")

# Extract text from a document
from stream_utils_text import extract_text, guess_title_and_date, naive_summary
text = extract_text(uploaded_file)
title, date = guess_title_and_date(text, filename)
summary = naive_summary(text)

# Insert a paper
from stream_utils_db import insert_paper
paper_id = insert_paper(
    conn,
    title=title,
    date_iso=date,
    section="A",
    subject="research",
    content=text,
    summary=summary
)

# Search papers
from stream_utils_db import fetch_papers
papers = fetch_papers(
    conn,
    date_from="2024-01-01",
    date_to="2024-12-31",
    section="A",
    subject="research"
)
```

## Module Documentation

### Table of Contents

1. [Storage S3 Module](#storage-s3-module)
2. [Stream App Module](#stream-app-module)
3. [Stream Metrics Module](#stream-metrics-module)
4. [Stream Utils DB Module](#stream-utils-db-module)
5. [Stream Utils Text Module](#stream-utils-text-module)
6. [Stream Voice Module](#stream-voice-module)
7. [Tracking MLflow Module](#tracking-mlflow-module)

---

## Storage S3 Module

**File:** `storage_s3.py`

AWS S3 integration for file storage and retrieval with automatic configuration management.

### Configuration

The module automatically detects S3 configuration from:
1. Streamlit secrets (`st.secrets`)
2. Environment variables

Required configuration:
- `S3_BUCKET` - S3 bucket name
- `AWS_REGION` - AWS region (optional)
- `S3_PREFIX` - Key prefix for organized storage (optional)

### Public Functions

#### `is_enabled() -> bool`
Check if S3 is properly configured and available.

**Returns:**
- `bool` - True if S3 is configured and boto3 is available

**Example:**
```python
from storage_s3 import is_enabled
if is_enabled():
    print("S3 storage is available")
```

#### `put_file(local_path: str, key: str) -> bool`
Upload a local file to S3.

**Parameters:**
- `local_path` (str) - Path to the local file
- `key` (str) - S3 object key (will be prefixed if S3_PREFIX is set)

**Returns:**
- `bool` - True if upload successful, False otherwise

**Example:**
```python
from storage_s3 import put_file
success = put_file("/path/to/document.pdf", "documents/research_paper.pdf")
if success:
    print("File uploaded successfully")
```

#### `get_file(key: str, local_path: str) -> bool`
Download a file from S3 to local storage.

**Parameters:**
- `key` (str) - S3 object key
- `local_path` (str) - Local destination path

**Returns:**
- `bool` - True if download successful, False otherwise

**Example:**
```python
from storage_s3 import get_file
success = get_file("documents/research_paper.pdf", "/tmp/downloaded_paper.pdf")
```

#### `exists(key: str) -> bool`
Check if an object exists in S3.

**Parameters:**
- `key` (str) - S3 object key

**Returns:**
- `bool` - True if object exists, False otherwise

**Example:**
```python
from storage_s3 import exists
if exists("documents/research_paper.pdf"):
    print("File exists in S3")
```

#### `url(key: str) -> Optional[str]`
Generate a presigned URL for an S3 object.

**Parameters:**
- `key` (str) - S3 object key

**Returns:**
- `Optional[str]` - Presigned URL (1 hour expiry) or None if failed

**Example:**
```python
from storage_s3 import url
presigned_url = url("documents/research_paper.pdf")
if presigned_url:
    print(f"Download URL: {presigned_url}")
```

---

## Stream App Module

**File:** `stream_app.py`

Main Streamlit application providing the complete Papers DB interface with multiple tabs for different functionalities.

### Key Features

- **Document Upload & Ingestion** - PDF/DOCX file processing with automatic text extraction
- **Semantic Search** - Vector-based document retrieval using Sentence Transformers
- **Weekly Notes Generation** - Automated summary generation for date ranges
- **QA and Metrics** - Evaluation tools for summarization and retrieval quality
- **Voice Notes** - Audio recording and transcription capabilities
- **Video Notes** - Manual note creation and management
- **Metrics Dashboard** - Performance tracking and trend analysis

### Public Functions

#### `load_models() -> Pipeline`
Load and cache the sentiment analysis model.

**Returns:**
- `Pipeline` - Hugging Face transformers pipeline for sentiment analysis

**Note:** This function is cached with `@st.cache_resource` for performance.

#### `s3_client() -> S3Client`
Get a cached S3 client instance.

**Returns:**
- `S3Client` - Boto3 S3 client

**Note:** This function is cached with `@st.cache_resource` for performance.

#### `build_embed_index(texts: List[str]) -> Dict[str, Any]`
Build a semantic search index from text documents.

**Parameters:**
- `texts` (List[str]) - List of text documents to index

**Returns:**
- `Dict[str, Any]` - Index containing embeddings and metadata

**Example:**
```python
texts = ["Document 1 content", "Document 2 content"]
index = build_embed_index(texts)
```

#### `cosine_topk(query: str, index: Dict[str, Any], k: int = 5) -> List[Tuple[int, float]]`
Perform semantic search using cosine similarity.

**Parameters:**
- `query` (str) - Search query
- `index` (Dict[str, Any]) - Pre-built embedding index
- `k` (int) - Number of top results to return (default: 5)

**Returns:**
- `List[Tuple[int, float]]` - List of (document_index, similarity_score) tuples

**Example:**
```python
results = cosine_topk("machine learning research", index, k=10)
for doc_idx, score in results:
    print(f"Document {doc_idx}: {score:.3f}")
```

#### `log_metrics_run(run_name: str, params: dict | None = None, metrics: dict | None = None, artifacts: dict[str, bytes] | None = None, artifact_path: str = "") -> None`
Log experiment metrics and artifacts to MLflow.

**Parameters:**
- `run_name` (str) - Name for the MLflow run
- `params` (dict, optional) - Parameters to log
- `metrics` (dict, optional) - Metrics to log
- `artifacts` (dict[str, bytes], optional) - Artifacts as filename -> bytes mapping
- `artifact_path` (str) - Path within the run for artifacts

**Example:**
```python
log_metrics_run(
    run_name="experiment_1",
    params={"model": "bert-base", "k": 10},
    metrics={"accuracy": 0.95, "f1": 0.92},
    artifacts={"results.csv": csv_bytes}
)
```

### UI Components

The application provides 7 main tabs:

1. **Upload** - Document ingestion with metadata assignment
2. **Query** - Database queries with filtering options
3. **Weekly Notes** - Automated summary generation for date ranges
4. **QA and Metrics** - Evaluation tools and semantic search
5. **Video Notes** - Manual note creation
6. **Voice Notes** - Audio recording and transcription
7. **Metrics** - Performance dashboard and trend analysis

---

## Stream Metrics Module

**File:** `stream_metrics.py`

Comprehensive evaluation metrics for information retrieval and text summarization tasks.

### Retrieval Metrics

#### `precision_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float`
Calculate precision at k for ranked retrieval results.

**Parameters:**
- `ranked_ids` (List[int]) - List of document IDs in ranked order
- `relevant_ids` (Set[int]) - Set of relevant document IDs
- `k` (int) - Cutoff point for evaluation

**Returns:**
- `float` - Precision at k (0.0 to 1.0)

**Example:**
```python
from stream_metrics import precision_at_k
ranked = [1, 3, 5, 2, 4]
relevant = {1, 2, 5}
p_at_3 = precision_at_k(ranked, relevant, k=3)  # 2/3 = 0.667
```

#### `recall_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float`
Calculate recall at k for ranked retrieval results.

**Parameters:**
- `ranked_ids` (List[int]) - List of document IDs in ranked order
- `relevant_ids` (Set[int]) - Set of relevant document IDs
- `k` (int) - Cutoff point for evaluation

**Returns:**
- `float` - Recall at k (0.0 to 1.0)

**Example:**
```python
from stream_metrics import recall_at_k
ranked = [1, 3, 5, 2, 4]
relevant = {1, 2, 5}
r_at_3 = recall_at_k(ranked, relevant, k=3)  # 2/3 = 0.667
```

#### `mrr(ranked_ids: List[int], relevant_ids: Set[int]) -> float`
Calculate Mean Reciprocal Rank (MRR).

**Parameters:**
- `ranked_ids` (List[int]) - List of document IDs in ranked order
- `relevant_ids` (Set[int]) - Set of relevant document IDs

**Returns:**
- `float` - MRR (0.0 to 1.0)

**Example:**
```python
from stream_metrics import mrr
ranked = [1, 3, 5, 2, 4]
relevant = {1, 2, 5}
mrr_score = mrr(ranked, relevant)  # 1/1 = 1.0 (first relevant at position 1)
```

#### `ndcg_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float`
Calculate Normalized Discounted Cumulative Gain at k.

**Parameters:**
- `ranked_ids` (List[int]) - List of document IDs in ranked order
- `relevant_ids` (Set[int]) - Set of relevant document IDs
- `k` (int) - Cutoff point for evaluation

**Returns:**
- `float` - NDCG at k (0.0 to 1.0)

**Example:**
```python
from stream_metrics import ndcg_at_k
ranked = [1, 3, 5, 2, 4]
relevant = {1, 2, 5}
ndcg = ndcg_at_k(ranked, relevant, k=5)
```

### Summarization Metrics

#### `rouge_l(candidate: str, reference: str) -> float`
Calculate ROUGE-L F1 score using Longest Common Subsequence.

**Parameters:**
- `candidate` (str) - Generated summary text
- `reference` (str) - Reference summary text

**Returns:**
- `float` - ROUGE-L F1 score (0.0 to 1.0)

**Example:**
```python
from stream_metrics import rouge_l
candidate = "The study shows machine learning improves accuracy."
reference = "Machine learning techniques increase model accuracy significantly."
score = rouge_l(candidate, reference)
```

#### `bertscore_optional(candidate: str, reference: str) -> Optional[Dict[str, float]]`
Calculate BERTScore metrics if available.

**Parameters:**
- `candidate` (str) - Generated summary text
- `reference` (str) - Reference summary text

**Returns:**
- `Optional[Dict[str, float]]` - Dictionary with 'bertscore_p', 'bertscore_r', 'bertscore_f1' or None if bert-score not installed

**Example:**
```python
from stream_metrics import bertscore_optional
result = bertscore_optional(candidate, reference)
if result:
    print(f"BERTScore F1: {result['bertscore_f1']:.3f}")
```

#### `entity_hallucination_report(summary: str, source: str) -> Dict[str, Any]`
Detect potential entity hallucinations in summaries.

**Parameters:**
- `summary` (str) - Generated summary text
- `source` (str) - Source document text

**Returns:**
- `Dict[str, Any]` - Report containing:
  - `summary_entities` - List of entities found in summary
  - `missing_in_source` - Entities in summary but not in source
  - `entity_coverage_rate` - Coverage rate (0.0 to 1.0)
  - `backend` - "spacy" or "regex" indicating detection method

**Example:**
```python
from stream_metrics import entity_hallucination_report
report = entity_hallucination_report(summary, source)
print(f"Entity coverage: {report['entity_coverage_rate']:.2f}")
print(f"Missing entities: {report['missing_in_source']}")
```

---

## Stream Utils DB Module

**File:** `stream_utils_db.py`

Database operations, schema management, and data persistence for the Papers DB system.

### Database Schema

The system uses SQLite with the following main tables:

#### Papers Table
```sql
CREATE TABLE papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    date_iso TEXT,            -- 'YYYY-MM-DD' or 'unknown'
    section TEXT,             -- 'A' | 'B' | 'C'
    subject TEXT,
    content TEXT,             -- full text
    summary TEXT,             -- cached summary
    s3_raw_key TEXT,          -- optional: original blob S3 key
    s3_text_key TEXT,         -- optional: extracted text S3 key
    created_at TEXT DEFAULT (datetime('now'))
);
```

#### Notes Table
```sql
CREATE TABLE notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    date_iso TEXT,
    content TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

#### Metrics Tables
- `metrics_runs` - Experiment run metadata
- `metrics_summary` - Summarization evaluation metrics
- `metrics_retrieval` - Retrieval evaluation metrics

### Core Functions

#### `init_db(path: str) -> sqlite3.Connection`
Initialize the database with all required tables and indexes.

**Parameters:**
- `path` (str) - Path to SQLite database file

**Returns:**
- `sqlite3.Connection` - Configured database connection

**Features:**
- Creates all tables and indexes
- Enables foreign keys and WAL mode
- Sets up FTS5 virtual table for full-text search
- Handles schema migrations automatically

**Example:**
```python
from stream_utils_db import init_db
conn = init_db("papers.db")
```

#### `insert_paper(conn: sqlite3.Connection, *, title: str, date_iso: str, section: str, subject: str, content: str, summary: str, s3_raw_key: str | None = None, s3_text_key: str | None = None) -> int`
Insert a new paper into the database.

**Parameters:**
- `conn` (sqlite3.Connection) - Database connection
- `title` (str) - Paper title
- `date_iso` (str) - Date in ISO format (YYYY-MM-DD) or 'unknown'
- `section` (str) - Section classification ('A', 'B', or 'C')
- `subject` (str) - Subject/topic classification
- `content` (str) - Full text content
- `summary` (str) - Generated summary
- `s3_raw_key` (str, optional) - S3 key for original file
- `s3_text_key` (str, optional) - S3 key for extracted text

**Returns:**
- `int` - ID of the inserted paper

**Example:**
```python
from stream_utils_db import insert_paper
paper_id = insert_paper(
    conn,
    title="Machine Learning in Healthcare",
    date_iso="2024-01-15",
    section="A",
    subject="healthcare",
    content="Full paper content...",
    summary="This paper discusses..."
)
```

#### `insert_note(conn: sqlite3.Connection, *, title: str, date_iso: str, content: str) -> int`
Insert a new note into the database.

**Parameters:**
- `conn` (sqlite3.Connection) - Database connection
- `title` (str) - Note title
- `date_iso` (str) - Date in ISO format
- `content` (str) - Note content

**Returns:**
- `int` - ID of the inserted note

#### `fetch_papers(conn: sqlite3.Connection, *, date_from: Optional[str], date_to: Optional[str], section: Optional[str], subject: Optional[str], title_contains: Optional[str], content_contains: Optional[str] = None) -> pd.DataFrame`
Query papers with flexible filtering options.

**Parameters:**
- `conn` (sqlite3.Connection) - Database connection
- `date_from` (Optional[str]) - Start date filter (ISO format)
- `date_to` (Optional[str]) - End date filter (ISO format)
- `section` (Optional[str]) - Section filter ('A', 'B', 'C')
- `subject` (Optional[str]) - Subject filter (partial match)
- `title_contains` (Optional[str]) - Title filter (partial match)
- `content_contains` (Optional[str]) - Content filter (uses FTS5 if available)

**Returns:**
- `pd.DataFrame` - Query results with all paper fields

**Features:**
- Uses FTS5 for content search when available
- Falls back to LIKE queries for compatibility
- Excludes 'unknown' dates from date range filters
- Results ordered by date (newest first) then ID

**Example:**
```python
from stream_utils_db import fetch_papers
papers = fetch_papers(
    conn,
    date_from="2024-01-01",
    date_to="2024-12-31",
    section="A",
    subject="research",
    content_contains="machine learning"
)
```

#### `get_distinct_subjects(conn: sqlite3.Connection) -> list[str]`
Get all distinct subjects from the papers table.

**Parameters:**
- `conn` (sqlite3.Connection) - Database connection

**Returns:**
- `list[str]` - Sorted list of unique subjects

### Metrics Functions

#### `ensure_metrics_tables(conn: sqlite3.Connection) -> None`
Ensure metrics tables exist in the database.

#### `start_metrics_run(conn: sqlite3.Connection, kind: str, notes: str = "") -> int`
Start a new metrics evaluation run.

**Parameters:**
- `conn` (sqlite3.Connection) - Database connection
- `kind` (str) - Type of evaluation ('summary' or 'retrieval')
- `notes` (str) - Optional notes for the run

**Returns:**
- `int` - Run ID for logging metrics

#### `log_summary_metric(conn: sqlite3.Connection, run_id: int, row_id: int, rouge_l: float, bert_p: float|None, bert_r: float|None, bert_f1: float|None, entity_coverage: float|None) -> None`
Log summarization evaluation metrics.

#### `log_retrieval_metric(conn: sqlite3.Connection, run_id: int, query: str, k: int, mode: str, precision: float, recall: float, mrr: float, ndcg: float) -> None`
Log retrieval evaluation metrics.

#### `latest_metrics_summary_view(conn: sqlite3.Connection) -> pd.DataFrame`
Get the latest summarization metrics with paper details.

#### `latest_metrics_retrieval_view(conn: sqlite3.Connection) -> pd.DataFrame`
Get the latest retrieval metrics.

---

## Stream Utils Text Module

**File:** `stream_utils_text.py`

Text extraction, processing, and summarization utilities supporting multiple document formats.

### Text Extraction

#### `extract_text(uploaded_file) -> str`
Extract text from uploaded documents with robust fallback handling.

**Parameters:**
- `uploaded_file` - Streamlit UploadedFile object

**Returns:**
- `str` - Extracted text content

**Supported Formats:**
- **PDF**: Uses pdfplumber (preferred) with pypdf fallback
- **DOCX**: Uses python-docx (preferred) with docx2txt fallback
- **TXT**: Direct text decoding with encoding detection

**Features:**
- Automatic format detection from filename
- Multiple extraction libraries for reliability
- Safe encoding detection and fallback
- Handles corrupted or malformed files gracefully

**Example:**
```python
from stream_utils_text import extract_text
text = extract_text(uploaded_file)
print(f"Extracted {len(text)} characters")
```

### Title and Date Extraction

#### `guess_title_and_date(text: str, filename: str = "") -> Tuple[str | None, str | None]`
Extract title and date from document text and filename.

**Parameters:**
- `text` (str) - Document text content
- `filename` (str) - Original filename (optional)

**Returns:**
- `Tuple[str | None, str | None]` - (title, iso_date) tuple

**Title Extraction:**
1. Uses filename stem (without extension)
2. Falls back to first non-empty line (2-120 characters)

**Date Extraction:**
1. Searches text content first
2. Falls back to filename
3. Supports multiple formats:
   - ISO: YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
   - US: MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
   - Month: "January 15, 2024", "Jan 15, 2024"

**Example:**
```python
from stream_utils_text import guess_title_and_date
title, date = guess_title_and_date(text, "research_paper_2024-01-15.pdf")
print(f"Title: {title}, Date: {date}")
```

### Text Summarization

#### `naive_summary(text: str, limit: int = 1200) -> str`
Generate a simple text summary by truncation and whitespace normalization.

**Parameters:**
- `text` (str) - Input text to summarize
- `limit` (int) - Maximum character limit (default: 1200)

**Returns:**
- `str` - Summarized text with "..." if truncated

**Features:**
- Normalizes whitespace (multiple spaces → single space)
- Truncates to specified limit
- Adds "..." indicator for truncated text
- Handles empty or None input gracefully

**Example:**
```python
from stream_utils_text import naive_summary
long_text = "This is a very long document with lots of content..."
summary = naive_summary(long_text, limit=500)
print(summary)  # "This is a very long document with lots of content..."
```

---

## Stream Voice Module

**File:** `stream_voice.py`

Audio processing and speech-to-text transcription capabilities with multiple backend support.

### Audio Processing

#### `_bytes_to_wav_16k_mono(audio_bytes: bytes, filename: str | None = None) -> bytes`
Normalize audio to mono 16kHz WAV format for transcription.

**Parameters:**
- `audio_bytes` (bytes) - Raw audio data
- `filename` (str, optional) - Original filename for format detection

**Returns:**
- `bytes` - Normalized WAV audio data

**Features:**
- Supports multiple input formats (MP3, M4A, WAV, etc.)
- Converts to mono 16kHz 16-bit PCM
- Uses pydub (preferred) or soundfile fallback
- Handles format detection from filename

### Transcription

#### `transcribe_bytes(audio_bytes: bytes, filename: str | None = None) -> str`
Transcribe audio to text using available transcription backends.

**Parameters:**
- `audio_bytes` (bytes) - Raw audio data
- `filename` (str, optional) - Original filename

**Returns:**
- `str` - Transcribed text

**Backend Priority:**
1. **faster-whisper** - Fast Whisper implementation (preferred)
2. **Vosk** - Offline fallback (requires VOSK_MODEL_PATH env var)

**Features:**
- Automatic audio normalization
- Multiple transcription engines
- Offline capability with Vosk
- Handles transcription failures gracefully

**Example:**
```python
from stream_voice import transcribe_bytes
with open("audio.wav", "rb") as f:
    audio_data = f.read()
transcript = transcribe_bytes(audio_data, "audio.wav")
print(transcript)
```

### Audio Recording

#### `record_or_upload_audio() -> Optional[bytes]`
Record audio from microphone or upload audio file.

**Returns:**
- `Optional[bytes]` - Raw audio data or None

**Features:**
- **Primary**: streamlit-mic-recorder for browser-based recording
- **Fallback**: File uploader for pre-recorded audio
- Supports WAV, MP3, M4A formats
- Returns raw bytes for processing

**Example:**
```python
from stream_voice import record_or_upload_audio
audio_bytes = record_or_upload_audio()
if audio_bytes:
    transcript = transcribe_bytes(audio_bytes)
    print(f"Transcribed: {transcript}")
```

### Backend-Specific Functions

#### `_transcribe_faster_whisper(wav_bytes: bytes) -> Optional[str]`
Transcribe using faster-whisper (CPU-optimized Whisper).

**Parameters:**
- `wav_bytes` (bytes) - Normalized WAV audio data

**Returns:**
- `Optional[str]` - Transcribed text or None if failed

**Configuration:**
- Uses "base" model (change to "small" for better quality)
- CPU-only execution for compatibility
- Temporary file handling for model input

#### `_transcribe_vosk(wav_bytes: bytes) -> Optional[str]`
Transcribe using Vosk (offline fallback).

**Parameters:**
- `wav_bytes` (bytes) - Normalized WAV audio data

**Returns:**
- `Optional[str]` - Transcribed text or None if failed

**Requirements:**
- VOSK_MODEL_PATH environment variable
- Vosk model files downloaded locally
- 16kHz mono WAV input format

---

## Tracking MLflow Module

**File:** `tracking_mlflow.py`

MLflow integration for experiment tracking, model management, and artifact logging.

### Configuration

The module automatically detects MLflow configuration from:
1. Streamlit secrets (`st.secrets`)
2. Environment variables

Required configuration:
- `MLFLOW_TRACKING_URI` - MLflow server URI
- `MLFLOW_EXPERIMENT` - Experiment name (optional, defaults to "default")

### Core Functions

#### `is_enabled() -> bool`
Check if MLflow is properly configured and available.

**Returns:**
- `bool` - True if MLflow is configured and importable

**Example:**
```python
from tracking_mlflow import is_enabled
if is_enabled():
    print("MLflow tracking is available")
```

#### `start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None)`
Context manager for MLflow runs.

**Parameters:**
- `run_name` (Optional[str]) - Name for the MLflow run
- `tags` (Optional[Dict[str, str]]) - Tags to associate with the run

**Returns:**
- Context manager yielding the active run (or None if disabled)

**Example:**
```python
from tracking_mlflow import start_run, log_params, log_metrics

with start_run("experiment_1", tags={"model": "bert", "dataset": "papers"}):
    log_params({"learning_rate": 0.001, "batch_size": 32})
    log_metrics({"accuracy": 0.95, "loss": 0.05})
```

#### `log_params(params: Dict[str, Any]) -> None`
Log parameters to the current MLflow run.

**Parameters:**
- `params` (Dict[str, Any]) - Parameters to log (converted to strings)

**Example:**
```python
from tracking_mlflow import log_params
log_params({
    "model_type": "transformer",
    "max_length": 512,
    "learning_rate": 0.0001
})
```

#### `log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None`
Log metrics to the current MLflow run.

**Parameters:**
- `metrics` (Dict[str, float]) - Metrics to log
- `step` (Optional[int]) - Step number for time-series metrics

**Example:**
```python
from tracking_mlflow import log_metrics
log_metrics({
    "precision": 0.92,
    "recall": 0.89,
    "f1_score": 0.90
}, step=100)
```

#### `log_artifact_file(local_path: str, artifact_path: Optional[str] = None) -> None`
Log a file as an artifact to the current MLflow run.

**Parameters:**
- `local_path` (str) - Path to the local file to log
- `artifact_path` (Optional[str]) - Path within the run for the artifact

**Example:**
```python
from tracking_mlflow import log_artifact_file
log_artifact_file("results.csv", "evaluation_results")
log_artifact_file("model.pkl", "models")
```

### Usage Patterns

#### Complete Experiment Tracking
```python
from tracking_mlflow import start_run, log_params, log_metrics, log_artifact_file
import json

# Start experiment
with start_run("summarization_eval", tags={"task": "summarization", "model": "bert"}):
    # Log parameters
    log_params({
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "temperature": 0.7
    })
    
    # Run evaluation
    results = evaluate_model()
    
    # Log metrics
    log_metrics({
        "rouge_l": results["rouge_l"],
        "bert_f1": results["bert_f1"],
        "entity_coverage": results["entity_coverage"]
    })
    
    # Log artifacts
    with open("detailed_results.json", "w") as f:
        json.dump(results, f)
    log_artifact_file("detailed_results.json", "evaluation")
```

#### Graceful Degradation
```python
from tracking_mlflow import is_enabled, start_run, log_metrics

# Check if MLflow is available
if is_enabled():
    with start_run("experiment"):
        log_metrics({"accuracy": 0.95})
else:
    print("MLflow not available, skipping tracking")
```

---

## Usage Examples

### Complete Document Processing Pipeline

```python
import streamlit as st
from stream_utils_db import init_db, insert_paper, fetch_papers
from stream_utils_text import extract_text, guess_title_and_date, naive_summary
from stream_metrics import rouge_l, bertscore_optional, entity_hallucination_report
from tracking_mlflow import start_run, log_metrics

# Initialize database
conn = init_db("papers.db")

# Process uploaded document
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
if uploaded_file:
    # Extract text
    text = extract_text(uploaded_file)
    
    # Extract metadata
    title, date = guess_title_and_date(text, uploaded_file.name)
    
    # Generate summary
    summary = naive_summary(text)
    
    # Insert into database
    paper_id = insert_paper(
        conn,
        title=title or "Unknown",
        date_iso=date or "unknown",
        section="A",
        subject="research",
        content=text,
        summary=summary
    )
    
    # Evaluate summary quality
    with start_run("summary_evaluation"):
        rouge_score = rouge_l(summary, text)
        bert_scores = bertscore_optional(summary, text)
        hallucination_report = entity_hallucination_report(summary, text)
        
        metrics = {"rouge_l": rouge_score}
        if bert_scores:
            metrics.update(bert_scores)
        metrics["entity_coverage"] = hallucination_report["entity_coverage_rate"]
        
        log_metrics(metrics)
    
    st.success(f"Processed paper {paper_id} with ROUGE-L: {rouge_score:.3f}")
```

### Semantic Search Implementation

```python
from stream_app import build_embed_index, cosine_topk
from stream_utils_db import fetch_papers

# Get all papers
papers_df = fetch_papers(conn, date_from=None, date_to=None, section=None, 
                        subject=None, title_contains=None, content_contains=None)

# Build search index
texts = (papers_df["title"].fillna("") + " " + papers_df["summary"].fillna("")).tolist()
index = build_embed_index(texts)

# Perform search
query = "machine learning healthcare applications"
results = cosine_topk(query, index, k=10)

# Display results
for doc_idx, score in results:
    paper = papers_df.iloc[doc_idx]
    st.write(f"**{paper['title']}** (Score: {score:.3f})")
    st.write(f"Section: {paper['section']}, Subject: {paper['subject']}")
    st.write(f"Summary: {paper['summary'][:200]}...")
```

### Metrics Evaluation Workflow

```python
from stream_utils_db import start_metrics_run, log_summary_metric, log_retrieval_metric
from stream_metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k

# Start evaluation run
run_id = start_metrics_run(conn, kind="retrieval", notes="gold_standard_eval")

# Evaluate retrieval performance
for query, relevant_ids in test_queries.items():
    # Get ranked results
    results = cosine_topk(query, index, k=10)
    ranked_ids = [papers_df.iloc[idx]["id"] for idx, _ in results]
    
    # Calculate metrics
    p_at_5 = precision_at_k(ranked_ids, relevant_ids, k=5)
    r_at_5 = recall_at_k(ranked_ids, relevant_ids, k=5)
    mrr_score = mrr(ranked_ids, relevant_ids)
    ndcg_score = ndcg_at_k(ranked_ids, relevant_ids, k=10)
    
    # Log metrics
    log_retrieval_metric(conn, run_id, query, k=5, mode="gold", 
                        precision=p_at_5, recall=r_at_5, mrr=mrr_score, ndcg=ndcg_score)
```

### Voice Note Processing

```python
from stream_voice import record_or_upload_audio, transcribe_bytes
from stream_utils_db import insert_paper

# Record or upload audio
audio_bytes = record_or_upload_audio()

if audio_bytes:
    # Transcribe audio
    transcript = transcribe_bytes(audio_bytes)
    
    if transcript:
        # Create paper entry
        paper_id = insert_paper(
            conn,
            title="Voice Note",
            date_iso="2024-01-15",
            section="C",
            subject="voice-note",
            content=transcript,
            summary=naive_summary(transcript)
        )
        
        st.success(f"Voice note processed as paper {paper_id}")
        st.text_area("Transcript", transcript)
```

## Configuration

### Environment Variables

```bash
# S3 Configuration
S3_BUCKET=your-bucket-name
AWS_REGION=us-west-2
S3_PREFIX=documents/

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=papers-db

# Vosk Configuration (for offline transcription)
VOSK_MODEL_PATH=/path/to/vosk/model
```

### Streamlit Secrets

Create `.streamlit/secrets.toml`:

```toml
[S3]
BUCKET = "your-bucket-name"
REGION = "us-west-2"
PREFIX = "documents/"

[MLFLOW]
TRACKING_URI = "http://localhost:5000"
EXPERIMENT = "papers-db"
```

## Dependencies

### Core Dependencies
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `sqlite3` - Database (built-in)

### Optional Dependencies
- `boto3` - AWS S3 integration
- `mlflow` - Experiment tracking
- `sentence-transformers` - Semantic search
- `transformers` - NLP models
- `pdfplumber` / `pypdf` - PDF processing
- `python-docx` / `docx2txt` - DOCX processing
- `faster-whisper` / `vosk` - Speech transcription
- `pydub` / `soundfile` - Audio processing
- `bert-score` - BERT evaluation metrics
- `spacy` - Named entity recognition
- `streamlit-mic-recorder` - Audio recording

## Error Handling

All modules implement graceful error handling:

- **Optional Dependencies**: Functions return None or empty results when optional libraries are unavailable
- **File Processing**: Multiple fallback methods for text extraction
- **Audio Processing**: Graceful degradation when transcription backends fail
- **Database Operations**: Proper error handling with rollback on failures
- **S3 Operations**: Return boolean success indicators instead of raising exceptions

## Performance Considerations

- **Caching**: Streamlit `@st.cache_resource` for expensive operations
- **Database**: WAL mode and proper indexing for concurrent access
- **Embeddings**: CPU-only execution to avoid CUDA issues
- **Audio**: Efficient resampling and format conversion
- **FTS**: SQLite FTS5 for fast full-text search

## Security Notes

- **S3**: Uses presigned URLs with 1-hour expiry
- **Database**: SQLite with parameterized queries to prevent injection
- **File Upload**: Validates file types and handles malicious uploads
- **Audio**: Temporary file cleanup after processing
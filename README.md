Introduction

This project lets you upload DOCX or PDF files, review them, and get clear summaries. It uses a retrieval step to find the most relevant passages and a small summarizer to write the final answer. The goal is simple. Make it easy to search documents, read the right parts, and save what you learned.

Key Features

Upload DOCX and PDF

Store texts and metadata in a local database

Retrieve top passages for any question

Generate short, focused summaries

Track retrieval and summary metrics

Optional storage in S3 for files and artifacts

Optional MLflow tracking for experiments

How Retrieval Augmented Generation works with LangChain

The app breaks each document into chunks. Each chunk gets an embedding. When you ask a question, the app finds the closest chunks and sends only those to the summarizer. LangChain ties these steps together so the same pipeline works in the app and in scripts.

What this gives you

Less noise because the model reads only relevant text

Faster responses on long files

Better answers on multi document searches

How RAG works in this project

Ingest
The app reads DOCX or PDF, extracts text, and records title, source, and timestamps.

Chunk
The text is split into small, overlapping pieces. This keeps context while limiting size.

Embed
Each chunk is turned into a vector and stored in the index.

Retrieve
For a query, the app finds the top k chunks by similarity.

Generate
The summarizer writes an answer using only those chunks. Citations point back to the sources.

Evaluate
The app can log retrieval metrics and summary metrics so you can see quality over time.

LangChain and S3 storage

LangChain wires together ingestion, embeddings, retrieval, and generation

Local SQLite holds metadata and metrics

If you provide AWS credentials, uploads and artifacts can be written to an S3 bucket

Paths and bucket names are set in environment variables so you can change them without code edits

Project workflow and results

Typical flow

Upload a few DOCX or PDF files

Click ingest to parse and chunk them

Ask a question or request a summary

Review the answer, the supporting passages, and the metrics

Save results to the database and optionally to S3

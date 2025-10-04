from __future__ import annotations
import sqlite3
from typing import Optional, List
import pandas as pd

# ================= Core & Migrations =================

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
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

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    date_iso TEXT,
    content TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(date_iso);
CREATE INDEX IF NOT EXISTS idx_papers_section ON papers(section);
CREATE INDEX IF NOT EXISTS idx_papers_subject ON papers(subject);
"""

# FTS5 virtual table + triggers to mirror 'papers'
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
  title, subject, summary, content,
  content='papers', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
  INSERT INTO papers_fts(rowid, title, subject, summary, content)
  VALUES (new.id, new.title, new.subject, new.summary, new.content);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, title, subject, summary, content)
  VALUES ('delete', old.id, old.title, old.subject, old.summary, old.content);
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, title, subject, summary, content)
  VALUES ('delete', old.id, old.title, old.subject, old.summary, old.content);
  INSERT INTO papers_fts(rowid, title, subject, summary, content)
  VALUES (new.id, new.title, new.subject, new.summary, new.content);
END;
"""

# Metrics tables + helpful indexes
METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT DEFAULT (datetime('now')),
    kind TEXT,          -- 'summary' or 'retrieval'
    notes TEXT
);

CREATE TABLE IF NOT EXISTS metrics_summary (
    run_id INTEGER,
    row_id INTEGER,
    rouge_l REAL,
    bert_p REAL,
    bert_r REAL,
    bert_f1 REAL,
    entity_coverage REAL,
    FOREIGN KEY(run_id) REFERENCES metrics_runs(id)
);

CREATE TABLE IF NOT EXISTS metrics_retrieval (
    run_id INTEGER,
    query TEXT,
    k INTEGER,
    mode TEXT,          -- 'gold' or 'proxy'
    precision REAL,
    recall REAL,
    mrr REAL,
    ndcg REAL,
    FOREIGN KEY(run_id) REFERENCES metrics_runs(id)
);

-- Helpful indexes for fast history/trends
CREATE INDEX IF NOT EXISTS idx_metrics_summary_run ON metrics_summary(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_retrieval_run ON metrics_retrieval(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_runs_time ON metrics_runs(run_at);
"""

def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    names = {r[1] for r in rows}  # (cid, name, type, notnull, dflt_value, pk)
    return col in names

def _ensure_papers_columns(conn: sqlite3.Connection) -> None:
    # Add S3 columns if table existed before this version
    for col in ("s3_raw_key", "s3_text_key"):
        if not _column_exists(conn, "papers", col):
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col} TEXT;")

def _fts_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts_check USING fts5(x);")
        conn.execute("DROP TABLE IF EXISTS __fts_check;")
        return True
    except Exception:
        return False

def _ensure_fts(conn: sqlite3.Connection) -> bool:
    if not _fts_available(conn):
        return False
    conn.executescript(FTS_SCHEMA)
    # Initial backfill if empty
    cnt = conn.execute("SELECT count(*) FROM papers_fts;").fetchone()[0]
    if cnt == 0:
        conn.execute("""
            INSERT INTO papers_fts(rowid, title, subject, summary, content)
            SELECT id, title, subject, summary, content FROM papers;
        """)
    return True

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Safety/perf pragmas
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    conn.executescript(SCHEMA)
    _ensure_papers_columns(conn)     # adds s3_raw_key / s3_text_key if missing
    conn.executescript(METRICS_SCHEMA)
    try:
        _ensure_fts(conn)            # creates papers_fts + triggers if FTS5 is available
    except Exception:
        pass
    conn.commit()
    return conn

# ================= Insert helpers =================

def insert_paper(conn: sqlite3.Connection, *, title: str, date_iso: str, section: str,
                 subject: str, content: str, summary: str,
                 s3_raw_key: str | None = None, s3_text_key: str | None = None) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO papers(title, date_iso, section, subject, content, summary, s3_raw_key, s3_text_key) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (title, date_iso, section, subject, content, summary, s3_raw_key, s3_text_key)
    )
    conn.commit()
    return cur.lastrowid

def insert_note(conn: sqlite3.Connection, *, title: str, date_iso: str, content: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notes(title, date_iso, content) VALUES (?,?,?)",
        (title, date_iso, content)
    )
    conn.commit()
    return cur.lastrowid

# ================= Query helpers =================

def fetch_papers(conn: sqlite3.Connection, *, 
                 date_from: Optional[str],
                 date_to: Optional[str],
                 section: Optional[str],
                 subject: Optional[str],
                 title_contains: Optional[str],
                 content_contains: Optional[str] = None) -> pd.DataFrame:
    """
    Returns rows from papers with flexible filters.
    Uses FTS5 for content/summary/subject search if available; falls back to LIKE otherwise.
    """
    where: List[str] = []
    params: List[str] = []

    # Date range (exclude 'unknown' from comparisons)
    if date_from:
        where.append("(date_iso!='unknown' AND date_iso>=?)"); params.append(date_from)
    if date_to:
        where.append("(date_iso!='unknown' AND date_iso<=?)"); params.append(date_to)

    if section:
        where.append("section=?"); params.append(section)
    if subject:
        where.append("subject LIKE ?"); params.append(f"%{subject}%")
    if title_contains:
        where.append("title LIKE ?"); params.append(f"%{title_contains}%")

    use_fts = False
    if content_contains:
        # Prefer FTS when available
        try:
            conn.execute("SELECT 1 FROM papers_fts LIMIT 1;")
            use_fts = True
        except Exception:
            use_fts = False

    base_select = """
    SELECT id, title, date_iso, section, subject, summary, content, s3_raw_key, s3_text_key
    FROM papers
    """

    if content_contains and use_fts:
        # Use MATCH; wrap in quotes to emulate 'contains' for simple terms
        # For advanced queries, the caller could pass full FTS syntax.
        match_q = f"\"{content_contains}\""
        where.append("id IN (SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?)")
        params.append(match_q)
    elif content_contains:
        like = f"%{content_contains}%"
        where.append("(content LIKE ? OR summary LIKE ? OR subject LIKE ?)")
        params.extend([like, like, like])

    sql = base_select
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(NULLIF(date_iso,'unknown'),'0000-00-00') DESC, id DESC"

    return pd.read_sql_query(sql, conn, params=params)

def get_distinct_subjects(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT subject FROM papers ORDER BY subject").fetchall()
    return [r[0] for r in rows if r[0]]

# ================= Metrics helpers =================

def ensure_metrics_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(METRICS_SCHEMA)
    conn.commit()

def start_metrics_run(conn: sqlite3.Connection, kind: str, notes: str = "") -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO metrics_runs(kind, notes) VALUES(?,?)", (kind, notes))
    conn.commit()
    return cur.lastrowid

def log_summary_metric(conn: sqlite3.Connection, run_id: int, row_id: int,
                       rouge_l: float, bert_p: float|None, bert_r: float|None,
                       bert_f1: float|None, entity_coverage: float|None) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO metrics_summary(run_id,row_id,rouge_l,bert_p,bert_r,bert_f1,entity_coverage)"
        " VALUES (?,?,?,?,?,?,?)",
        (run_id, row_id, rouge_l, bert_p, bert_r, bert_f1, entity_coverage)
    )
    conn.commit()

def log_retrieval_metric(conn: sqlite3.Connection, run_id: int, query: str, k: int,
                         mode: str, precision: float, recall: float, mrr: float, ndcg: float) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO metrics_retrieval(run_id,query,k,mode,precision,recall,mrr,ndcg)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (run_id, query, k, mode, precision, recall, mrr, ndcg)
    )
    conn.commit()

def latest_metrics_summary_view(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT ms.run_id, mr.run_at, ms.row_id, p.title, p.date_iso, p.section, p.subject,
           ms.rouge_l, ms.bert_p, ms.bert_r, ms.bert_f1, ms.entity_coverage
    FROM metrics_summary ms
    JOIN metrics_runs mr ON mr.id = ms.run_id
    JOIN papers p ON p.id = ms.row_id
    ORDER BY mr.run_at DESC, ms.row_id DESC
    """
    return pd.read_sql_query(sql, conn)

def latest_metrics_retrieval_view(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT mr.id AS run_id, mr.run_at, r.query, r.k, r.mode, r.precision, r.recall, r.mrr, r.ndcg
    FROM metrics_retrieval r
    JOIN metrics_runs mr ON mr.id = r.run_id
    ORDER BY mr.run_at DESC, r.rowid DESC
    """
    return pd.read_sql_query(sql, conn)

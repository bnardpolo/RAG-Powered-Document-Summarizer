

from __future__ import annotations
import os
import io
import tempfile
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Papers DB", layout="wide")

# env
load_dotenv()

st.title("My App (sanity)")

@st.cache_resource
def load_models():
    # import heavy libs here so they don't run on module import
    from transformers import pipeline
    return pipeline("sentiment-analysis")

@st.cache_resource
def s3_client():
    import boto3
    return boto3.client("s3")

if st.button("Load model"):
    nlp = load_models()
    st.success("Model ready!")

# ---------------- Local modules ----------------
from stream_utils_db import (
    init_db,
    insert_paper,
    fetch_papers,
    get_distinct_subjects,
    insert_note,
    ensure_metrics_tables,
    start_metrics_run,
    log_summary_metric,
    log_retrieval_metric,
    latest_metrics_summary_view,
    latest_metrics_retrieval_view,
)
from stream_utils_text import extract_text, guess_title_and_date, naive_summary
from stream_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    rouge_l,
    bertscore_optional,
    entity_hallucination_report,
)
from stream_voice import record_or_upload_audio, transcribe_bytes

@st.cache_resource(show_spinner=False)
def _get_embedder():
    # Force CPU to avoid CUDA/meta-tensor issues
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def build_embed_index(texts: List[str]) -> Dict[str, Any]:
    model = _get_embedder()
    if not texts:
        return {"emb": np.empty((0, 384), dtype=np.float32)}
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device="cpu",
        show_progress_bar=False,
    )
    emb = emb.astype(np.float32, copy=False)
    return {"emb": emb}

def cosine_topk(query: str, index: Dict[str, Any], k: int = 5) -> List[Tuple[int, float]]:
    emb = index.get("emb")
    if emb is None or emb.size == 0:
        return []
    model = _get_embedder()
    q = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        device="cpu",
        show_progress_bar=False,
    )[0].astype(np.float32, copy=False)
    sims = emb @ q  # cosine since normalized
    k = max(1, min(k, sims.shape[0]))
    top = np.argpartition(-sims, np.arange(k))[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]

def _mlflow_setup():
    try:
        import mlflow
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        exp = os.getenv("MLFLOW_EXPERIMENT", "papers-app")
        mlflow.set_experiment(exp)
        return mlflow
    except Exception as e:
        print("[mlflow] setup skipped:", e)
        return None

def log_metrics_run(run_name: str,
                    params: dict | None = None,
                    metrics: dict | None = None,
                    artifacts: dict[str, bytes] | None = None,
                    artifact_path: str = "") -> None:
    mlflow = _mlflow_setup()
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            if metrics:
                clean = {}
                for k, v in metrics.items():
                    try:
                        if v is None:
                            continue
                        clean[k] = float(v)
                    except Exception:
                        pass
                if clean:
                    mlflow.log_metrics(clean)
            if artifacts:
                for fname, data in artifacts.items():
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(data)
                        tmp.flush()
                        mlflow.log_artifact(tmp.name, artifact_path=artifact_path or None)
    except Exception as e:
        print("[mlflow] logging skipped:", e)


st.write("Papers DB")

DB_PATH = "papers.db"
conn = init_db(DB_PATH)
ensure_metrics_tables(conn)


with st.sidebar:
    st.write("Metrics")
    auto_eval_after_ingest = st.checkbox("Auto-evaluate summaries after ingest", value=True)
    proxy_retrieval_on = st.checkbox("Enable proxy retrieval metrics if no labels", value=True)

tabs = st.tabs(
    [
        "Upload",
        "Query",
        "Weekly Notes",
        "QA and Metrics",
        "Video Notes",
        "Voice Notes",
        "Metrics",
    ]
)


with tabs[0]:
    st.write("Upload Papers")
    files = st.file_uploader(
        "Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        section = st.selectbox("Section", ["A", "B", "C"], index=0)
    with col2:
        subject = st.text_input("Subject", value="general")
    with col3:
        use_override_date = st.checkbox("Enable override date")
        override_date = (
            st.date_input("Override date", value=dt.date.today())
            if use_override_date else None
        )

    if st.button("Ingest"):
        rows = []
        if files:
            for f in files:
                text = extract_text(f)
                title, found_date = guess_title_and_date(text, f.name)
                title = title or "unknown"
                date_iso = (
                    override_date.isoformat()
                    if isinstance(override_date, dt.date)
                    else (found_date or "unknown")
                )
                summary = naive_summary(text)
                insert_paper(
                    conn,
                    title=title,
                    date_iso=date_iso,
                    section=section,
                    subject=subject,
                    content=text,
                    summary=summary,
                )
                rows.append(
                    {"file": f.name, "title": title, "date": date_iso, "section": section, "subject": subject}
                )

            st.write(pd.DataFrame(rows))

            # Per-date summary downloads
            all_lines = []
            if rows:
                dates = sorted({r["date"] for r in rows})
                for d in dates:
                    if d != "unknown":
                        df_d = fetch_papers(
                            conn,
                            date_from=d,
                            date_to=d,
                            section=None,
                            subject=None,
                            title_contains=None,
                            content_contains=None,
                        )
                    else:
                        df_all = fetch_papers(
                            conn,
                            date_from=None, date_to=None, section=None,
                            subject=None, title_contains=None, content_contains=None
                        )
                        titles_unknown = {r["title"] for r in rows if r["date"] == "unknown"}
                        df_d = df_all[df_all["title"].isin(titles_unknown)]

                    lines = [f"Date: {d}", ""]
                    for _, rr in df_d.iterrows():
                        lines.append(f"- [{rr['section']}] {rr['subject']} :: {rr['title']}")
                        lines.append((rr["summary"] or "")[:1000])
                        lines.append("")
                    blob = "\n".join(lines).encode("utf-8")
                    st.download_button(
                        f"Download summary for {d}",
                        data=blob,
                        file_name=f"summary_{(d.replace('-','_') if d!='unknown' else 'unknown')}.txt",
                        mime="text/plain",
                        key=f"dl_sum_{d}"
                    )
                    all_lines.extend(lines + ["", ""])

                if all_lines:
                    st.download_button(
                        "Download all new-date summaries",
                        data="\n".join(all_lines).encode("utf-8"),
                        file_name="summaries_all_dates.txt",
                        mime="text/plain",
                        key="dl_sum_all_dates"
                    )

           
            if auto_eval_after_ingest and rows:
                titles = [r["title"] for r in rows]
                df_new = fetch_papers(conn, date_from=None, date_to=None, section=None,
                                      subject=None, title_contains=None, content_contains=None)
                df_new = df_new[df_new["title"].isin(titles)]
                run_id = start_metrics_run(conn, kind="summary", notes="post-ingest summary eval")
                per_row = []
                for _, rr in df_new.iterrows():
                    source = rr["content"] or ""
                    summ = rr["summary"] or ""
                    rl = rouge_l(summ, source)
                    bs = bertscore_optional(summ, source)
                    if bs is None:
                        bp = br = bf = None
                    else:
                        bp, br, bf = bs.get("bertscore_p"), bs.get("bertscore_r"), bs.get("bertscore_f1")
                    hall = entity_hallucination_report(summ, source)
                    cov = hall.get("entity_coverage_rate") if isinstance(hall, dict) else None
                    log_summary_metric(conn, run_id, int(rr["id"]), float(rl), bp, br, bf, cov)
                    per_row.append({"id": int(rr["id"]), "rouge_l": rl, "bert_p": bp, "bert_r": br, "bert_f1": bf, "entity_coverage": cov})
                st.write("Auto-evaluated summaries for new rows.")

                
                try:
                    dfm = pd.DataFrame(per_row)
                    avg = {}
                    if not dfm.empty:
                        avg = {
                            "rouge_l": float(dfm["rouge_l"].mean()),
                            "bert_p": float(dfm["bert_p"].dropna().mean()) if dfm["bert_p"].notna().any() else None,
                            "bert_r": float(dfm["bert_r"].dropna().mean()) if dfm["bert_r"].notna().any() else None,
                            "bert_f1": float(dfm["bert_f1"].dropna().mean()) if dfm["bert_f1"].notna().any() else None,
                            "entity_coverage": float(dfm["entity_coverage"].dropna().mean()) if dfm["entity_coverage"].notna().any() else None,
                        }
                    artifacts = {}
                    if all_lines:
                        artifacts["summaries_all_dates.txt"] = "\n".join(all_lines).encode("utf-8")
                    log_metrics_run(
                        run_name=f"summary_auto_run_{run_id}",
                        params={"kind":"summary","notes":"post-ingest"},
                        metrics={k:v for k,v in avg.items() if v is not None},
                        artifacts=artifacts,
                        artifact_path="ingest_summaries"
                    )
                except Exception as _e:
                    print("[mlflow] upload auto-eval logging skipped:", _e)
        else:
            st.write("No files selected")

with tabs[1]:
    st.write("Query")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        enable_from = st.checkbox("Enable Date from")
        d_from = st.date_input("Date from", value=dt.date.today()) if enable_from else None
    with c2:
        enable_to = st.checkbox("Enable Date to")
        d_to = st.date_input("Date to", value=dt.date.today()) if enable_to else None
    with c3:
        sec = st.selectbox("Section filter", ["Any", "A", "B", "C"], index=0)
    with c4:
        subs = ["Any"] + get_distinct_subjects(conn)
        sub = st.selectbox("Subject filter", subs, index=0)

    doc_contains = st.text_input("Document contains (content, summary, or subject)")
    title_q = st.text_input("Title contains")

    if st.button("Run query"):
        df = fetch_papers(
            conn,
            date_from=d_from.isoformat() if isinstance(d_from, dt.date) else None,
            date_to=d_to.isoformat() if isinstance(d_to, dt.date) else None,
            section=None if sec == "Any" else sec,
            subject=None if sub == "Any" else sub,
            title_contains=title_q or None,
            content_contains=doc_contains or None,
        )
        st.write(df)
        if not df.empty:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "query.csv",
                "text/csv",
            )

with tabs[2]:
    st.write("Weekly Notes")
    ws = st.date_input("Week start")
    we = st.date_input("Week end")
    section_w = st.selectbox("Section filter", ["Any", "A", "B", "C"], index=0, key="wk_sec")
    subject_w = st.text_input("Subject contains", value="", key="wk_sub")

    if st.button("Build weekly notes"):
        dfw = fetch_papers(
            conn,
            date_from=ws.isoformat() if isinstance(ws, dt.date) else None,
            date_to=we.isoformat() if isinstance(we, dt.date) else None,
            section=None if section_w == "Any" else section_w,
            subject=subject_w or None,
            title_contains=None,
            content_contains=None,
        )
        if dfw.empty:
            st.write("No items")
        else:
            bullets = []
            for _, r in dfw.iterrows():
                bullets.append(f"- ({r['date_iso']}) [{r['section']}] {r['subject']}: {r['title']}")
            digest = "Weekly Notes\n\n" + "\n".join(bullets)
            st.text_area("Notes", digest, height=240)
            st.download_button(
                "Download Notes",
                digest.encode("utf-8"),
                "weekly_notes.txt",
                "text/plain"
            )

# ---------------- QA and Metrics ----------------
with tabs[3]:
    st.write("QA and Metrics")
    st.write("Embedding retrieval (Sentence-Transformers) over titles and summaries.")

    df_all = fetch_papers(
        conn, date_from=None, date_to=None, section=None, subject=None, title_contains=None, content_contains=None
    )

    enable_retrieval = st.checkbox(
        "Enable retrieval engine (downloads model on first use)",
        value=False
    )

    if enable_retrieval:
        texts = (df_all["title"].fillna("") + " " + df_all["summary"].fillna("")).tolist()
        ids = df_all["id"].tolist()

        vec = build_embed_index(texts)   # heavy: runs only if user enables
        if vec["emb"].shape[0] == 0:
            st.info("No usable documents yet. Ingest papers with non-empty text to enable retrieval and metrics.")
        else:
            q = st.text_input("Query text")
            k = st.number_input("Top k", min_value=1, max_value=50, value=5, step=1, key="k_ret")

            if st.button("Retrieve"):
                topk = cosine_topk(q, vec, k=int(k))
                show = []
                for idx, score in topk:
                    rid = ids[idx]
                    row = df_all[df_all["id"] == rid].iloc[0]
                    show.append(
                        {
                            "id": rid,
                            "score": float(score),
                            "title": row["title"],
                            "section": row["section"],
                            "date": row["date_iso"],
                        }
                    )
                st.write(pd.DataFrame(show))

            st.write("Retrieval metrics need gold labels. If none provided, a proxy can be computed.")
            gold_csv = st.text_area("Paste labeled CSV (optional)")

            if st.button("Score retrieval now"):
                from io import StringIO
                if gold_csv.strip():
                    run_id = start_metrics_run(conn, kind="retrieval", notes="gold")
                    gold = pd.read_csv(StringIO(gold_csv))
                    for _, r in gold.iterrows():
                        q_ = str(r["query"])
                        rel = set(map(int, str(r["relevant_ids"]).split("|"))) if str(r["relevant_ids"]).strip() else set()
                        topk = cosine_topk(q_, vec, k=int(k))
                        ranked_ids = [ids[idx] for idx, _ in topk]
                        p = precision_at_k(ranked_ids, rel, int(k))
                        rc = recall_at_k(ranked_ids, rel, int(k))
                        rr_ = mrr(ranked_ids, rel)
                        nd = ndcg_at_k(ranked_ids, rel, int(k))
                        log_retrieval_metric(conn, run_id, q_, int(k), "gold", p, rc, rr_, nd)
                    st.write("Logged retrieval metrics with gold labels.")

                    # MLflow logging (gold)
                    try:
                        df_ret_latest = latest_metrics_retrieval_view(conn)
                        df_this = df_ret_latest[df_ret_latest["run_id"] == run_id]
                        avg = {}
                        if not df_this.empty:
                            avg = {
                                "precision": float(df_this["precision"].mean()),
                                "recall": float(df_this["recall"].mean()),
                                "mrr": float(df_this["mrr"].mean()),
                                "ndcg": float(df_this["ndcg"].mean()),
                            }
                        csv_bytes = df_this.to_csv(index=False).encode("utf-8") if not df_this.empty else b""
                        log_metrics_run(
                            run_name=f"retrieval_gold_run_{run_id}",
                            params={"kind":"retrieval","mode":"gold","k": int(k)},
                            metrics=avg,
                            artifacts={"retrieval_gold_results.csv": csv_bytes},
                            artifact_path="retrieval_eval"
                        )
                    except Exception as _e:
                        print("[mlflow] qa retrieval gold logging skipped:", _e)

                elif proxy_retrieval_on:
                    run_id = start_metrics_run(conn, kind="retrieval", notes="proxy")
                    proxy_queries = list({(t or "")[:60] for t in df_all["title"].tolist()})[:10]
                    for q_ in proxy_queries:
                        topk = cosine_topk(q_, vec, k=int(k))
                        ranked_ids = [ids[idx] for idx, _ in topk]
                        q_low = q_.lower()
                        rel = set()
                        for _, row in df_all.iterrows():
                            blob = f"{row['title']} {row['summary']} {row['subject']}".lower()
                            if any(tok for tok in q_low.split() if tok and tok in blob):
                                rel.add(int(row["id"]))

                        p = precision_at_k(ranked_ids, rel, int(k))
                        rc = recall_at_k(ranked_ids, rel, int(k))
                        rr_ = mrr(ranked_ids, rel)
                        nd = ndcg_at_k(ranked_ids, rel, int(k))
                        log_retrieval_metric(conn, run_id, q_, int(k), "proxy", p, rc, rr_, nd)
                    st.write("Logged retrieval metrics in proxy mode.")

                    # MLflow logging (proxy)
                    try:
                        df_ret_latest = latest_metrics_retrieval_view(conn)
                        df_this = df_ret_latest[df_ret_latest["run_id"] == run_id]
                        avg = {}
                        if not df_this.empty:
                            avg = {
                                "precision": float(df_this["precision"].mean()),
                                "recall": float(df_this["recall"].mean()),
                                "mrr": float(df_this["mrr"].mean()),
                                "ndcg": float(df_this["ndcg"].mean()),
                            }
                        csv_bytes = df_this.to_csv(index=False).encode("utf-8") if not df_this.empty else b""
                        log_metrics_run(
                            run_name=f"retrieval_proxy_run_{run_id}",
                            params={"kind":"retrieval","mode":"proxy","k": int(k)},
                            metrics=avg,
                            artifacts={"retrieval_proxy_results.csv": csv_bytes},
                            artifact_path="retrieval_eval"
                        )
                    except Exception as _e:
                        print("[mlflow] qa retrieval proxy logging skipped:", _e)
                else:
                    st.write("No gold labels and proxy mode disabled. Nothing evaluated.")
    else:
        st.caption("Retrieval disabled to keep startup fast.")

    # Metrics quick view
    st.write("Metrics History")
    if st.checkbox("Show latest summary metrics"):
        st.write(latest_metrics_summary_view(conn))
    if st.checkbox("Show latest retrieval metrics"):
        st.write(latest_metrics_retrieval_view(conn))

# ---------------- Video Notes ----------------
with tabs[4]:
    st.write("Video Notes")
    st.write("Enter a dated note. After saving you can file it to a Section.")
    note_date = st.date_input("Date", value=dt.date.today())
    note_title = st.text_input("Note title")
    note_text = st.text_area("Note text")
    if st.button("Save note"):
        tmp_id = insert_note(
            conn,
            title=note_title.strip() or "untitled",
            date_iso=note_date.isoformat(),
            content=note_text.strip(),
        )
        st.write(f"Saved temporary note id {tmp_id}")
        sec_choice = st.selectbox("Choose Section", ["A", "B", "C"], key=f"sec_note_{tmp_id}")
        if st.button("File note into section"):
            insert_paper(
                conn,
                title=note_title.strip() or "untitled",
                date_iso=note_date.isoformat(),
                section=sec_choice,
                subject="video-note",
                content=note_text.strip(),
                summary=naive_summary(note_text.strip()),
            )
            st.write("Filed")

# ---------------- Voice Notes ----------------
with tabs[5]:
    st.write("Voice Notes")
    audio_bytes = record_or_upload_audio()
    if audio_bytes and st.button("Transcribe"):
        text = transcribe_bytes(audio_bytes)
        if not text:
            st.write("No transcription produced")
        else:
            st.text_area("Transcript", text, height=200)
            date_ = st.date_input("Date", value=dt.date.today(), key="vn_date")
            sec_ = st.selectbox("Section", ["A", "B", "C"], key="vn_sec")
            title_ = st.text_input("Title", value="voice-note")
            if st.button("Save"):
                insert_paper(
                    conn,
                    title=title_.strip() or "voice-note",
                    date_iso=date_.isoformat(),
                    section=sec_,
                    subject="voice-note",
                    content=text,
                    summary=naive_summary(text),
                )
                st.write("Saved")

# ---------------- Metrics ----------------
with tabs[6]:
    st.write("Metrics")

    # Filters for history
    col_a, col_b = st.columns(2)
    with col_a:
        enable_start = st.checkbox("Enable start date")
        start_date = st.date_input("Start date") if enable_start else None
    with col_b:
        enable_end = st.checkbox("Enable end date")
        end_date = st.date_input("End date") if enable_end else None

    # Summarization metrics history
    st.write("Summarization metrics history")
    df_sum = latest_metrics_summary_view(conn)
    if not df_sum.empty:
        if start_date:
            df_sum = df_sum[df_sum["date_iso"].fillna("") >= start_date.isoformat()]
        if end_date:
            df_sum = df_sum[df_sum["date_iso"].fillna("") <= end_date.isoformat()]

        st.write("Rows")
        st.write(df_sum)

        st.download_button(
            "Download summary metrics CSV",
            df_sum.to_csv(index=False).encode("utf-8"),
            "metrics_summary_history.csv",
            "text/csv"
        )

        if len(df_sum) >= 2:
            st.write("Trends")
            df_plot = df_sum[["run_at","rouge_l","bert_p","bert_r","bert_f1","entity_coverage"]].copy()
            df_plot = df_plot.groupby("run_at").mean(numeric_only=True).reset_index()
            st.line_chart(df_plot.set_index("run_at")[["rouge_l"]])
            if "bert_f1" in df_plot.columns and df_plot["bert_f1"].notna().any():
                st.line_chart(df_plot.set_index("run_at")[["bert_p","bert_r","bert_f1"]])
            if "entity_coverage" in df_plot.columns and df_plot["entity_coverage"].notna().any():
                st.line_chart(df_plot.set_index("run_at")[["entity_coverage"]])
    else:
        st.write("No summary metrics logged yet.")

    # Retrieval metrics history
    st.write("Retrieval metrics history")
    df_ret = latest_metrics_retrieval_view(conn)
    if not df_ret.empty:
        if start_date:
            df_ret = df_ret[df_ret["run_at"] >= start_date.isoformat()]
        if end_date:
            df_ret = df_ret[df_ret["run_at"] <= end_date.isoformat()]

        st.write("Rows")
        st.write(df_ret)

        st.download_button(
            "Download retrieval metrics CSV",
            df_ret.to_csv(index=False).encode("utf-8"),
            "metrics_retrieval_history.csv",
            "text/csv"
        )

        grp = df_ret.groupby(["run_id","run_at","mode"], as_index=False)[["precision","recall","mrr","ndcg"]].mean(numeric_only=True)
        if len(grp) >= 1:
            st.write("Trends")
            for mode in grp["mode"].unique():
                st.write(f"{mode} averages")
                dfm = grp[grp["mode"] == mode].sort_values("run_at")
                st.line_chart(dfm.set_index("run_at")[["precision","recall","mrr","ndcg"]])
    else:
        st.write("No retrieval metrics logged yet.")

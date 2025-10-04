from __future__ import annotations
import io, re, datetime as dt
from typing import Tuple


def _safe_read(uploaded_file) -> bytes:
    # Streamlit's UploadedFile supports .getvalue() and .read()
    try:
        return uploaded_file.getvalue()
    except Exception:
        try:
            return uploaded_file.read()
        except Exception:
            return b""

def _safe_decode(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except Exception:
        try:
            import chardet  # optional but helpful
            enc = chardet.detect(b).get("encoding") or "utf-8"
            return b.decode(enc, "ignore")
        except Exception:
            return b.decode("latin-1", "ignore")

def _is_pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")

def _is_docx(name: str) -> bool:
    return name.lower().endswith(".docx")

def _is_txt(name: str) -> bool:
    return name.lower().endswith(".txt")


def extract_text(uploaded_file) -> str:
    """
    Robust extractor:
      - PDF: pdfplumber (preferred), fallback pypdf
      - DOCX: python-docx (preferred), fallback docx2txt
      - TXT/unknown: safe decode
    """
    name = getattr(uploaded_file, "name", "") or ""
    raw = _safe_read(uploaded_file)

    if _is_pdf(name):
        # Try pdfplumber (better layout)
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            if text:
                return text
        except Exception:
            pass
        # Fallback: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()
            if text:
                return text
        except Exception:
            pass
        # Last fallback: decode bytes
        return _safe_decode(raw).strip()

    if _is_docx(name):
        # Try python-docx
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join([p.text for p in doc.paragraphs]).strip()
        except Exception:
            pass
        # Fallback: docx2txt
        try:
            import docx2txt, tempfile, os
            with tempfile.TemporaryDirectory() as td:
                pth = f"{td}/t.docx"
                with open(pth, "wb") as f:
                    f.write(raw)
                text = docx2txt.process(pth) or ""
                return text.strip()
        except Exception:
            pass
        return _safe_decode(raw).strip()

    # TXT or unknown → safe decode
    if _is_txt(name) or True:
        return _safe_decode(raw).strip()



# ISO yyyy-mm-dd
_RE_ISO = re.compile(r"\b(20\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])\b")
# mm/dd/yyyy or mm-dd-yyyy
_RE_MDYY = re.compile(r"\b(0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])[-/\.](20\d{2})\b")
# Month dd, yyyy
_RE_MONTH = re.compile(r"\b([A-Za-z]{3,9})\s+(0?[1-9]|[12]\d|3[01]),?\s+(20\d{2})\b")

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July",
     "August","September","October","November","December"], start=1)
}

def _norm_date(groups) -> str | None:
    try:
        # yyyy, mm, dd
        if len(groups) == 3 and len(groups[0]) == 4:
            y, m, d = int(groups[0]), int(groups[1]), int(groups[2])
            return dt.date(y, m, d).isoformat()
    except Exception:
        pass
    try:
        # mm, dd, yyyy
        if len(groups) == 3 and len(groups[2]) == 4:
            m, d, y = int(groups[0]), int(groups[1]), int(groups[2])
            return dt.date(y, m, d).isoformat()
    except Exception:
        pass
    try:
        # Month, dd, yyyy
        if len(groups) == 3 and isinstance(groups[0], str):
            mon = _MONTHS.get(groups[0].lower())
            if mon:
                d = int(groups[1]); y = int(groups[2])
                return dt.date(y, mon, d).isoformat()
    except Exception:
        pass
    return None

def guess_title_and_date(text: str, filename: str = "") -> Tuple[str | None, str | None]:
    """
    Title: prefer filename stem; else first non-empty short-ish line (<=120 chars).
    Date: look inside text first, then filename. Return (title, iso_date or None).
    """
    # Title from filename stem
    title = None
    if filename:
        base = filename.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        title = stem.strip() or None
    # Fallback: first reasonable line from text
    if not title:
        for line in (text or "").splitlines():
            s = line.strip()
            if 2 <= len(s) <= 120:
                title = s
                break
    title = title or "unknown"

    # Date: text first
    for rx in (_RE_ISO, _RE_MDYY, _RE_MONTH):
        m = rx.search(text or "")
        if m:
            iso = _norm_date(m.groups())
            if iso:
                return title, iso
    # Then filename
    for rx in (_RE_ISO, _RE_MDYY, _RE_MONTH):
        m = rx.search(filename or "")
        if m:
            iso = _norm_date(m.groups())
            if iso:
                return title, iso
    return title, None

def naive_summary(text: str, limit: int = 1200) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    out = re.sub(r"\s+", " ", t)
    return out[:limit] + ("..." if len(out) > limit else "")

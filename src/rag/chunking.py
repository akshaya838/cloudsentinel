"""
Chunking utilities for the RAG pipeline.

CVE descriptions are short enough to embed as single chunks, but posture
findings and audit logs will often be longer, so this uses a simple sliding
window over words with overlap, which is enough for structured security text
without pulling in a heavier sentence-segmentation dependency.
"""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: input text.
        chunk_size: max words per chunk.
        overlap: words shared between consecutive chunks, so a fact that
            straddles a chunk boundary isn't lost entirely from either chunk.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def chunk_record(record: dict, source_type: str) -> list[dict]:
    """Chunk one CVE/posture/audit record into indexable pieces with metadata.

    Each chunk keeps a pointer back to its source record so retrieval results
    can cite exactly where an answer came from (source_id + source_type),
    which is what makes the RAG answers auditable rather than just plausible.
    """
    text = record.get("description") or record.get("text") or ""
    pieces = chunk_text(text)

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append({
            "chunk_id": f"{record.get('cve_id', record.get('id', 'unknown'))}::{i}",
            "text": piece,
            "source_id": record.get("cve_id", record.get("id", "unknown")),
            "source_type": source_type,
            "cvss_severity": record.get("cvss_severity"),
            "predicted_severity": (record.get("severity_prediction") or {}).get("predicted_severity"),
            "business_impact": record.get("business_impact", []),
        })
    return chunks

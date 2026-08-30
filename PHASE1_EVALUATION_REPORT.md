# CloudSentinel — Phase I Internal Evaluation

Covers rubric rows 11 (Data Collection & Preprocessing), 12 (1st NLP Concept: NER),
and 13 (2nd & 3rd NLP Concepts: Text Classification, Question Answering / RAG).

---

## Row 11 — Data Collection & Preprocessing

- **Source**: `src/ingestion/fetch_cve.py` — paginated client for the NVD REST API.
- **Dataset for this checkpoint**: `data/raw/cves_expanded.json` — 38 real, publicly
  documented CVEs (expanded from the original 5-record dev sample specifically so
  Phase I metrics reflect actual generalization, not memorization of a handful of examples).
- **Fields extracted per record**: CVE ID, description, CVSS base score, CVSS severity
  label, associated CWE IDs.
- **Class balance** (important — shows this isn't a toy single-class dataset):
  - CRITICAL: 17, HIGH: 11, MEDIUM: 6, LOW: 4
- **Preprocessing before downstream stages**: description text is passed as-is into
  spaCy's tokenizer (handles whitespace/punctuation normalization internally); CWE ID
  lists are normalized to the `CWE-NNN` string format used by the business-impact
  lookup table.
- **Data confidentiality note** (ties into the project's local-deployment pitch): CVE
  data is fully public, but the operational data CloudSentinel is designed for (cloud
  posture findings, audit logs) is sensitive — hence the local-Ollama deployment model.
  This checkpoint uses only public CVE data; posture/audit-log ingestion uses synthetic
  data by design.

**Path to real NVD pull**: `fetch_cve.py` already implements this; it wasn't used for
tonight's checkpoint because this sandboxed evaluation environment has no network route
to the NVD API. Running it from your own machine is a one-line command and is a natural
next step to mention as "already built, ready to scale past the curated sample."

---

## Row 12 — 1st NLP Concept: Named Entity Recognition

**Pipeline**: spaCy `en_core_web_sm` statistical NER + a custom `EntityRuler` layered on
top for security-specific entities (`CVE_ID`, `CWE_ID`, `VERSION_RANGE`,
`VENDOR_PRODUCT`, `ATTACK_VECTOR`) — a standard low-resource domain-adaptation pattern.

**Evaluation method**: 15 hand-labeled CVE descriptions (`src/ner/evaluate_ner.py`),
scored with strict exact-span-and-label matching (the standard used in NER benchmarks).

| Entity type | Precision | Recall | F1 |
|---|---|---|---|
| VERSION_RANGE | 1.00 | 1.00 | 1.00 |
| ATTACK_VECTOR | 1.00 | 0.889 | 0.941 |
| VENDOR_PRODUCT | 1.00 | 0.909 | 0.952 |
| **Overall (micro)** | **1.00** | **0.929** | **0.963** |

**Two real bugs found and fixed during this evaluation round** (good to mention live —
shows the evaluation process actually did its job):
1. Version numbers with more than 3 segments (e.g. Chrome's `116.0.5845.187`) and
   version numbers with an attached patch letter and no hyphen (e.g. OpenSSL's `1.0.1f`)
   were truncated by the original regex. Fixed by widening the numeric-segment and
   suffix-letter patterns in `ner_pipeline.py`.
2. The word "versions" was inconsistently included in the reported span depending on
   which regex branch matched, causing spurious mismatches against any downstream
   consumer expecting a clean version string. Fixed with a named capture group so the
   qualifier word is stripped for the "X through Y" case (kept for "prior to X" / "before
   X", where it's meaningful).

**Two known, documented limitations remaining** (honest talking point, not a hidden gap):
- `VENDOR_PRODUCT` requires an exact phrase match against a small hand-seeded dictionary
  — e.g. "Linux" alone isn't recognized unless written as "Linux kernel". The documented
  upgrade path (already noted in the codebase) is to source this from the CPE dictionary
  instead of hand-maintaining it.
- `ATTACK_VECTOR` is keyword-matched, so near-synonyms outside the seed list (e.g.
  "directory traversal" vs. the listed "path traversal") are missed. A fine-tuned
  NER model (the RoBERTa/DeBERTa upgrade already planned) would generalize past this.

---

## Row 13a — 2nd NLP Concept: Text Classification (Severity)

**Pipeline**: TF-IDF + Logistic Regression, wrapped with LIME so every prediction shows
which words drove it (model-agnostic — the wrapper won't need to change when this is
later swapped for a fine-tuned transformer).

**Evaluation method**: proper held-out test split (28 train / 10 test, stratified) on
the 38-record expanded set — not fit-and-score-on-the-same-5-records, which the
original code explicitly flagged as meaningless.

| Metric | Value |
|---|---|
| Test accuracy | 0.50 |
| Macro F1 | 0.33 |
| Weighted F1 | 0.40 |

**Honest framing for the evaluator**: 50% on a 10-example test split is exactly what
you'd expect from a TF-IDF+LogReg baseline trained on 28 examples — it's a real number,
not an inflated one, and it validates the *pipeline*, matching the caveat already
written into the code comments. The stated next step (already on your roadmap) is
scaling to a few hundred labeled examples and swapping in the fine-tuned
RoBERTa/DeBERTa classifier stub that's already sketched in `severity_classifier.py`.

**Business-impact tagging** (rule-based CWE → CIA-triad mapping — extended from 11 to
25 CWE codes to cover the expanded dataset): 38/38 records successfully mapped.

**Integration point to state explicitly**: NER's extracted `CWE_ID` entities are what
feed the business-impact tagger — this is the first integration hop in the pipeline.

---

## Row 13b — 3rd NLP Concept: Question Answering / RAG

**Pipeline**: word-based overlapping chunking → multilingual sentence-transformer
embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, chosen specifically so English and
Tamil queries land in the same embedding space) → ChromaDB persistent vector store →
top-k retrieval with full source-chunk citation.

**Evaluation method**: 24 test queries against known source CVEs, measuring
Hit-Rate@1, Hit-Rate@3, and Mean Reciprocal Rank (`src/rag/evaluate_retrieval.py`).
18 queries name the vulnerability directly ("What is Log4Shell?"); 6 are fully
paraphrased with no nickname or exact product name, to test semantic retrieval rather
than keyword overlap.

**Important caveat to state upfront tomorrow**: this sandboxed development environment
has no network route to huggingface.co, so the actual multilingual embedding model
couldn't be downloaded here tonight. The numbers below use a TF-IDF cosine-similarity
substitute to validate the chunking/indexing/retrieval *logic* end-to-end — run
`evaluate_retrieval.py` on your own machine (one command, ~30 seconds with internet) to
get the final embedding-based numbers before submission.

| Metric | Value (TF-IDF substitute) |
|---|---|
| Hit-Rate@1 | 23/24 = 0.958 |
| Hit-Rate@3 | 23/24 = 0.958 |
| MRR | 0.958 |

The one miss was a fully paraphrased query about the XZ Utils backdoor with no lexical
overlap with the source text — exactly the case a lexical method like TF-IDF can't
solve and semantic embeddings can. This is a genuinely useful result: it's direct
evidence for *why* the project uses sentence-transformer embeddings instead of simpler
keyword search, not just a design choice stated on faith.

**Integration point to state explicitly**: classified severity and business-impact tags
are stored as chunk metadata, so retrieval results carry not just the matching text but
its severity/impact context — this is the second integration hop.

---

## The one-breath integration narrative (for row 13's "integration" scoring)

> "A raw CVE description goes through NER, which pulls out the CVE ID, CWE IDs,
> affected versions, and attack vector. The extracted CWE IDs feed the business-impact
> tagger and the description feeds the severity classifier — both of those together
> attach severity and CIA-triad tags to the record. The record is then chunked and
> embedded, and its severity/impact tags travel with it as retrieval metadata into the
> vector index. A user's natural-language question then retrieves the right chunk *and*
> its severity context, with a citation back to the source CVE."

---

## Files changed / added tonight

- `data/raw/cves_expanded.json` — new, 38-record curated dataset
- `src/ner/ner_pipeline.py` — regex fixes (multi-segment versions, suffix letters, span drift)
- `src/ner/evaluate_ner.py` — new, NER evaluation script + gold set
- `src/classification/severity_classifier.py` — CWE→CIA map extended 11→25 entries
- `src/classification/evaluate_classifier.py` — new, train/test split evaluation
- `src/rag/evaluate_retrieval.py` — new, retrieval evaluation script + query set
- `data/eval/*.json` — saved metric outputs for all three concepts

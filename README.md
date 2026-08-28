# CloudSentinel

CloudSentinel is an NLP-powered assistant for cloud security incident triage and query resolution. It ingests CVE reports, cloud security posture findings, and audit logs, extracts structured entities, classifies severity and business impact, and answers natural-language security questions through a retrieval-augmented generation (RAG) pipeline. Every answer is grounded in cited source snippets and paired with remediation suggestions, rather than being generated as an unverifiable black box.

## Motivation

Security teams routinely triage a high volume of unstructured information: CVE advisories, cloud posture scan results, and audit logs, each with its own format and vocabulary. Existing commercial copilots (Microsoft Security Copilot, CrowdStrike Charlotte AI, Dropzone AI, and similar tools) address this space, but converge on the same claims of being vendor-agnostic and cloud-native. CloudSentinel is positioned differently, around two gaps that are largely unaddressed by current tools:

- **Multilingual support for English and Tamil**, covering both typed and voice input, aimed at Indian cloud and IT engineering teams who are not well served by English-only tooling.
- **A fully local-mode deployment option**, running on a self-hosted LLM with no dependency on a paid external API. This serves two purposes at once: it removes the cost barrier for small teams, and it keeps sensitive posture and audit data from ever leaving the organization's own infrastructure.

## Core capabilities

- Named entity recognition over CVE and security text (affected vendor/product, version ranges, CWE identifiers, attack vectors)
- Severity and business-impact classification with explainability (LIME) so a triage decision can be justified, not just asserted
- Retrieval-augmented Q&A over CVE, posture, and audit data, with every answer traceable back to its source snippet
- Remediation suggestions generated alongside each answer
- English and Tamil support, in both text and voice
- Fully containerized local-mode deployment with no required external API calls

## Architecture

```
CVE feeds / posture findings / audit logs
              |
              v
      Ingestion & cleaning
              |
              v
     NER (entity extraction)
              |
              v
  Severity / impact classifier
              |
              v
   Chunking + embedding + vector store
              |
              v
      RAG retrieval + local LLM
              |
              v
  Q&A interface (English / Tamil, text / voice)
```

## Tech stack

| Layer | Tools |
|---|---|
| NER | spaCy, HuggingFace Transformers |
| Classification | RoBERTa / DeBERTa, LIME |
| Embeddings | sentence-transformers (multilingual) |
| Vector store | ChromaDB or Qdrant |
| RAG orchestration | LangChain or LlamaIndex |
| Local LLM | Ollama (Llama 3.1 or Mistral) |
| Speech | Whisper (Tamil STT), Coqui TTS (Tamil TTS) |
| Backend | FastAPI |
| Frontend | Streamlit or React |
| Deployment | Docker |

## Project status

| Stage | Component | Status |
|---|---|---|
| 1 | Data layer: CVE ingestion and synthetic posture/audit data | In progress |
| 2 | NER pipeline | In progress |
| 3 | Severity / business-impact classifier | Baseline working, transformer upgrade pending |
| 4 | Vector store and RAG retrieval | Working (chunking, embedding, retrieval verified end to end) |
| 5 | Local LLM generation | Not started |
| 6 | FastAPI backend and UI | Not started |
| 7 | Tamil multilingual layer | Not started |
| 8 | Voice input/output | Not started |

## Repository structure

```
cloudsentinel/
    data/
        raw/            raw pulled CVE, posture, and audit data
        processed/      cleaned and chunked data ready for embedding
        synthetic/      synthetic posture findings and audit logs used for the demo
    src/
        ingestion/      CVE fetching and cleaning
        ner/            entity extraction pipeline
        classification/ severity and business-impact classifier
        rag/            chunking, embedding, vector store, retrieval
        api/            FastAPI backend
    docker/             Dockerfile and compose file for local-mode deployment
    tests/
    notebooks/          exploration and evaluation notebooks
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Stage 1: Fetch CVE data

```bash
python src/ingestion/fetch_cve.py --days 30 --severity HIGH --out data/raw/cves.json
```

This queries the public NVD API. No API key is required, but setting one as the `NVD_API_KEY` environment variable raises the rate limit considerably and is recommended for anything beyond small test pulls.

### Stage 2: Run NER on CVE descriptions

```bash
python src/ner/ner_pipeline.py --in data/raw/sample_cves.json --out data/processed/cves_ner.json
```

Extracts affected vendor/product, version ranges, CVE and CWE identifiers, and attack-vector keywords from each CVE description, combining spaCy's statistical NER with a rule-based layer tuned for security text.

### Stage 3: Classify severity and tag business impact

```bash
python src/classification/severity_classifier.py --in data/processed/cves_ner.json --out data/processed/cves_classified.json
```

Runs a TF-IDF plus Logistic Regression baseline severity classifier (LIME-explained, so each prediction shows the words that drove it) and a rule-based CWE-to-CIA-triad business-impact tagger. The baseline classifier is a deliberate placeholder: it needs a training set of a few hundred labeled examples or more, pulled with `fetch_cve.py`, before its accuracy is meaningful. The upgrade path to a fine-tuned RoBERTa/DeBERTa classifier is sketched directly in `src/classification/severity_classifier.py`.

### Stage 4: Build the vector index and retrieve

```bash
cd src/rag
python build_index.py --in ../../data/processed/cves_classified.json --collection cve_knowledge --persist-dir ../../data/processed/chroma_store
python retrieve.py --query "What is the impact of Log4Shell?" --top-k 3
```

`build_index.py` chunks each record, embeds the chunks with a multilingual sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`, chosen so English and Tamil share one embedding space for the later Tamil Q&A stage), and stores them in a persistent ChromaDB collection with severity and business-impact metadata attached. `retrieve.py` embeds a natural-language query and returns the top-k nearest chunks with their source CVE ID and metadata, which is the piece the local LLM generation stage will consume next.

The first run downloads the embedding model from HuggingFace, so it needs an internet connection once; every run after that is fully offline, in line with the project's local-mode design.

## Data and privacy

CVE data used in this project is fully public. Cloud posture findings and audit logs are treated as sensitive by design: the academic demo uses synthetic and anonymized data only, and the local-mode deployment path exists specifically so that, in a real deployment, this class of data never has to leave the organization's own infrastructure.

## Roadmap

The next component to implement is the severity and business-impact classifier, followed by the vector store and RAG retrieval layer. The Tamil multilingual layer and voice input/output are deliberately sequenced last, after the English pipeline is fully validated end to end.

## Academic context

This project is being developed as an academic submission, with the CVE-plus-Tamil-language angle intended as its primary novel contribution relative to existing commercial security copilots. Each pipeline component is chosen to map to specific prior work (CPE-Identifier and Few-Sample NER for entity extraction, RoBERTa/DeBERTa with LIME for severity classification, SERC and Wazuh-copilot style approaches for the RAG security assistant design, and MuRIL with the chaii-1 dataset for Tamil question answering), which the methodology section maps out in detail.

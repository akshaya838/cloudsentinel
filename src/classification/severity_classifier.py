"""
Severity and business-impact classification for CVE / security text.

Two components:

1. Severity classifier (baseline): TF-IDF + Logistic Regression trained on
   CVSS severity labels (LOW/MEDIUM/HIGH/CRITICAL), wrapped with LIME so every
   prediction can show which words drove it. This is a deliberate baseline —
   it needs a real training set (hundreds of labeled examples, not the 5-record
   sample) before its accuracy numbers mean anything. Once you have that,
   swap TF-IDF+LogReg for a fine-tuned RoBERTa/DeBERTa classifier
   (see `train_transformer_classifier` stub below) and keep the same LIME
   wrapper — LIME works on any predict_proba-style function, model-agnostic.

2. Business-impact tagger (rule-based): maps CWE identifiers to the CIA triad
   (Confidentiality / Integrity / Availability) using a small reference table.
   This is intentionally rule-based rather than learned, since there is no
   labeled business-impact dataset to train on yet — it's a placeholder that
   gives the RAG layer something structured to reason over until a proper
   classifier (or a larger CWE-to-impact mapping) replaces it.

Usage:
    python src/classification/severity_classifier.py \
        --in data/processed/cves_ner.json \
        --out data/processed/cves_classified.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Minimal CWE -> CIA-triad impact mapping. Extend this as the corpus grows;
# a fuller version should eventually be derived from the MITRE CWE catalog
# rather than hand-maintained.
CWE_IMPACT_MAP = {
    "CWE-20": ["Integrity"],                          # Improper input validation
    "CWE-59": ["Integrity"],                          # Link following
    "CWE-78": ["Confidentiality", "Integrity", "Availability"],  # OS command injection
    "CWE-79": ["Integrity"],                          # XSS
    "CWE-89": ["Confidentiality", "Integrity"],        # SQL injection
    "CWE-94": ["Confidentiality", "Integrity", "Availability"],  # Code injection
    "CWE-306": ["Confidentiality", "Integrity"],       # Missing authentication
    "CWE-400": ["Availability"],                       # Uncontrolled resource consumption
    "CWE-502": ["Confidentiality", "Integrity", "Availability"], # Insecure deserialization
    "CWE-506": ["Integrity"],                          # Embedded malicious code
    "CWE-787": ["Confidentiality", "Integrity", "Availability"], # Out-of-bounds write
}


def train_baseline_classifier(records: list[dict]):
    """Train the TF-IDF + Logistic Regression baseline on available records.

    WARNING: with only a handful of labeled records this will overfit and
    the reported behavior is for pipeline validation only, not accuracy.
    Re-run this once data/raw/cves.json has a few hundred+ labeled examples
    (see src/ingestion/fetch_cve.py).
    """
    texts = [r["description"] for r in records if r.get("cvss_severity")]
    labels = [r["cvss_severity"] for r in records if r.get("cvss_severity")]

    if len(set(labels)) < 2:
        raise ValueError(
            "Need at least 2 distinct severity classes in the training data. "
            "Pull a larger, more varied CVE sample with fetch_cve.py first."
        )

    pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipeline.fit(texts, labels)
    return pipeline


def explain_prediction(pipeline, text: str, num_features: int = 8) -> dict:
    """Return the words that most influenced the predicted severity class."""
    explainer = LimeTextExplainer(class_names=pipeline.classes_.tolist())

    exp = explainer.explain_instance(
        text,
        pipeline.predict_proba,
        num_features=num_features,
        labels=list(range(len(pipeline.classes_))),
    )

    predicted_idx = int(np.argmax(pipeline.predict_proba([text])[0]))
    predicted_label = pipeline.classes_[predicted_idx]

    contributing_words = [
        {"word": word, "weight": round(weight, 4)}
        for word, weight in exp.as_list(label=predicted_idx)
    ]

    return {
        "predicted_severity": predicted_label,
        "confidence": round(float(pipeline.predict_proba([text])[0][predicted_idx]), 4),
        "contributing_words": contributing_words,
    }


def tag_business_impact(cwe_ids: list[str]) -> list[str]:
    """Rule-based CIA-triad tagging from CWE identifiers."""
    impacts = set()
    for cwe in cwe_ids:
        impacts.update(CWE_IMPACT_MAP.get(cwe, []))
    return sorted(impacts) if impacts else ["Unclassified"]


def train_transformer_classifier(records: list[dict]):
    """Stub for the upgrade path: fine-tune RoBERTa/DeBERTa on severity labels
    once a real labeled dataset exists. Not implemented yet — this is stage 3b.

    Sketch of what this becomes:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=len(SEVERITY_LABELS)
        )
        # tokenize records, wrap in a Dataset, fine-tune with Trainer
        # LIME wrapping stays identical — swap pipeline.predict_proba for
        # a function that runs the fine-tuned model and returns softmax probs.
    """
    raise NotImplementedError(
        "Transformer fine-tuning requires a labeled dataset of a few hundred+ "
        "examples. Implement once data/raw/cves.json has been built out."
    )


def main():
    parser = argparse.ArgumentParser(description="Classify CVE severity and tag business impact.")
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    parser.add_argument("--explain-first-n", type=int, default=2,
                         help="Run LIME explanation on the first N records (slower).")
    args = parser.parse_args()

    records = json.loads(Path(args.in_path).read_text())

    print(f"Training baseline severity classifier on {len(records)} records...")
    print("NOTE: this is a small demo dataset. Treat predictions as a pipeline")
    print("check, not a real accuracy result, until you train on a larger corpus.\n")

    pipeline = train_baseline_classifier(records)

    for i, record in enumerate(records):
        record["business_impact"] = tag_business_impact(record.get("cwe_ids", []))

        if i < args.explain_first_n:
            explanation = explain_prediction(pipeline, record["description"])
            record["severity_prediction"] = explanation
        else:
            proba = pipeline.predict_proba([record["description"]])[0]
            predicted_idx = int(np.argmax(proba))
            record["severity_prediction"] = {
                "predicted_severity": pipeline.classes_[predicted_idx],
                "confidence": round(float(proba[predicted_idx]), 4),
            }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved {len(records)} classified records -> {out_path}")

    print("\nSample output:")
    sample = records[0]
    print(f"  {sample['cve_id']}")
    print(f"  actual severity:    {sample.get('cvss_severity')}")
    print(f"  predicted severity: {sample['severity_prediction']['predicted_severity']} "
          f"(confidence {sample['severity_prediction']['confidence']})")
    if "contributing_words" in sample["severity_prediction"]:
        print(f"  top contributing words: {sample['severity_prediction']['contributing_words']}")
    print(f"  business impact tags: {sample['business_impact']}")


if __name__ == "__main__":
    main()

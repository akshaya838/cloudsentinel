"""
Evaluate the severity classifier with a held-out test split.

Fitting and scoring on the same 5 records (the original demo sample) is
meaningless — it just measures memorization. This script trains on the
expanded 38-record curated CVE set (data/raw/cves_expanded.json) with a
stratified train/test split, so the reported accuracy/F1 reflect actual
generalization instead of overfitting to the training set.

Usage:
    python src/classification/evaluate_classifier.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from severity_classifier import SEVERITY_LABELS, tag_business_impact

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def main():
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "cves_expanded.json"
    records = json.loads(data_path.read_text())

    texts = [r["description"] for r in records]
    labels = [r["cvss_severity"] for r in records]

    print(f"Loaded {len(records)} records. Class distribution:")
    for label in SEVERITY_LABELS:
        print(f"  {label:<10} {labels.count(label)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print(f"Held-out test accuracy: {acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}   Weighted F1: {weighted_f1:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels_present = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("           " + "  ".join(f"{l:<10}" for l in labels_present))
    for i, row in enumerate(cm):
        print(f"{labels_present[i]:<10} " + "  ".join(f"{v:<10}" for v in row))

    # Business impact tagging sanity check (rule-based, evaluated separately —
    # this isn't a learned component so we just confirm coverage, not accuracy)
    tagged = [tag_business_impact(r.get("cwe_ids", [])) for r in records]
    unclassified = sum(1 for t in tagged if t == ["Unclassified"])
    print(f"\nBusiness impact tagging coverage: {len(records) - unclassified}/{len(records)} "
          f"records mapped to a CIA-triad tag ({unclassified} unclassified — CWEs not yet in the reference table).")

    results = {
        "n_train": len(X_train), "n_test": len(X_test),
        "test_accuracy": round(acc, 3),
        "macro_f1": round(macro_f1, 3),
        "weighted_f1": round(weighted_f1, 3),
        "class_distribution": {label: labels.count(label) for label in SEVERITY_LABELS},
        "business_impact_coverage": f"{len(records) - unclassified}/{len(records)}",
    }
    out_path = Path(__file__).parent.parent.parent / "data" / "eval" / "classifier_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

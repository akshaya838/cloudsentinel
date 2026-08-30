"""
Evaluate the NER pipeline against a small hand-labeled gold set.

This is the Phase-I "reliability, accuracy" checkpoint for the NER concept.
The gold set is 15 CVE descriptions pulled from data/raw/cves_expanded.json,
each hand-annotated with the entity spans a security analyst would expect to
see extracted (CVE_ID, CWE_ID, VERSION_RANGE, VENDOR_PRODUCT, ATTACK_VECTOR).

Scoring is exact span+label match (start, end, label) — the strictest
reasonable standard, matching how NER systems are typically benchmarked
(e.g. CoNLL-style exact-match scoring).

Usage:
    python src/ner/evaluate_ner.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ner_pipeline import build_pipeline, extract_entities

# --- Gold set --------------------------------------------------------------
# Each entry: (description text, [(surface_text, label), ...])
# Surface text is located in the description via str.find to get exact
# character offsets, so gold spans line up with however the text is written.
GOLD = [
    ("A remote code execution flaw in Apache Log4j2 versions 2.0-beta9 through 2.14.1. "
     "Attackers who control log messages or log message parameters can execute arbitrary "
     "code loaded from LDAP servers when message lookup substitution is enabled, commonly "
     "referred to as Log4Shell.",
     [("apache log4j2", "VENDOR_PRODUCT"), ("2.0-beta9 through 2.14.1", "VERSION_RANGE"),
      ("remote code execution", "ATTACK_VECTOR")]),

    ("A heap buffer overflow exists in the WebP image format handling in Google Chrome "
     "versions prior to 116.0.5845.187, allowing a remote attacker to potentially exploit "
     "heap corruption via a crafted HTML page.",
     [("google chrome", "VENDOR_PRODUCT"), ("prior to 116.0.5845.187", "VERSION_RANGE"),
      ("heap buffer overflow", "ATTACK_VECTOR")]),

    ("Spring Framework versions 5.3.0 through 5.3.17 and 5.2.0 through 5.2.19, when deployed "
     "on Apache Tomcat as a WAR with JDK 9 or higher, are vulnerable to remote code execution "
     "via data binding, an issue known as Spring4Shell.",
     [("spring framework", "VENDOR_PRODUCT"), ("apache tomcat", "VENDOR_PRODUCT"),
      ("5.3.0 through 5.3.17", "VERSION_RANGE"), ("5.2.0 through 5.2.19", "VERSION_RANGE"),
      ("remote code execution", "ATTACK_VECTOR")]),

    ("WinRAR versions prior to 6.23 allow attackers to execute arbitrary code when a user "
     "opens a specially crafted archive containing a file with a matching name but different "
     "extension, exploited via a Trojanized RAR/ZIP archive.",
     [("winrar", "VENDOR_PRODUCT"), ("prior to 6.23", "VERSION_RANGE")]),

    ("Malicious code was introduced into the upstream XZ Utils tarballs versions 5.6.0 and "
     "5.6.1 via build-system obfuscation, allowing a remote attacker with a specific private "
     "key to intercept and modify SSHD authentication on affected Linux distributions.",
     [("xz utils", "VENDOR_PRODUCT"), ("linux", "VENDOR_PRODUCT")]),

    ("OpenSSL versions 1.0.1 through 1.0.1f contain an out-of-bounds read in the TLS heartbeat "
     "extension, allowing remote attackers to read up to 64 kilobytes of process memory, "
     "including private keys and session data, commonly referred to as Heartbleed.",
     [("openssl", "VENDOR_PRODUCT"), ("1.0.1 through 1.0.1f", "VERSION_RANGE")]),

    ("The SMBv1 server in Microsoft Windows mishandles crafted packets, allowing remote "
     "attackers to execute arbitrary code via a buffer overflow, the vulnerability exploited "
     "by the EternalBlue tool and the WannaCry ransomware outbreak.",
     [("microsoft windows", "VENDOR_PRODUCT"), ("buffer overflow", "ATTACK_VECTOR")]),

    ("GNU Bash versions through 4.3 process trailing strings after function definitions in "
     "environment variables, allowing remote attackers to execute arbitrary code via a "
     "crafted environment, commonly referred to as Shellshock.",
     []),

    ("The Windows Print Spooler service improperly performs privileged file operations, "
     "allowing a remote authenticated attacker to execute arbitrary code with system "
     "privileges, an issue known as PrintNightmare.",
     []),  # no ATTACK_VECTOR keyword or known product literally appears — tests false-positive rate

    ("Citrix Application Delivery Controller and Gateway versions before 11.1.51.21 allow "
     "directory traversal, enabling an unauthenticated remote attacker to execute arbitrary "
     "code on the appliance.",
     [("directory traversal", "ATTACK_VECTOR"), ("before 11.1.51.21", "VERSION_RANGE")]),  # note: pipeline dict has "path traversal", not "directory traversal" — tests recall miss on near-synonym

    ("VMware vCenter Server contains a remote code execution vulnerability in the vSAN "
     "Health Check plug-in, exploitable via port 443 by an attacker with network access, "
     "due to insufficient input validation.",
     [("remote code execution", "ATTACK_VECTOR")]),

    ("Atlassian Confluence Server and Data Center are vulnerable to unauthenticated remote "
     "code execution via an OGNL injection flaw reachable through a crafted URL, affecting "
     "multiple supported versions.",
     [("remote code execution", "ATTACK_VECTOR")]),

    ("Progress MOVEit Transfer contains a SQL injection vulnerability in the web application, "
     "allowing an unauthenticated attacker to gain unauthorized access to the underlying "
     "database and escalate privileges.",
     [("sql injection", "ATTACK_VECTOR")]),

    ("Fortinet FortiOS SSL VPN web portal versions before 6.0.5 allow an unauthenticated "
     "attacker to download system files via specially crafted HTTP resource requests, an "
     "issue enabling path traversal.",
     [("fortinet fortios", "VENDOR_PRODUCT"), ("before 6.0.5", "VERSION_RANGE"),
      ("path traversal", "ATTACK_VECTOR")]),

    ("A flaw in the way the Linux kernel's pipe buffering mechanism handles page cache "
     "allows a local unprivileged user to overwrite data in read-only files, an issue known "
     "as Dirty Pipe.",
     [("linux kernel", "VENDOR_PRODUCT")]),
]


def span_match(gold_ents, pred_ents):
    """Return (tp, fp, fn) sets of (start, end, label) tuples for one example."""
    gold_set = {(e[0], e[1], e[2]) for e in gold_ents}
    pred_set = {(e["start"], e["end"], e["label"]) for e in pred_ents}
    tp = gold_set & pred_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set
    return tp, fp, fn


def resolve_gold_spans(text, annotations):
    """Convert (surface_text, label) annotations into (start, end, label) using str.find."""
    resolved = []
    for surface, label in annotations:
        idx = text.lower().find(surface.lower())
        if idx == -1:
            print(f"  WARNING: gold surface '{surface}' not found in text — skipping")
            continue
        resolved.append((idx, idx + len(surface), label))
    return resolved


def main():
    nlp = build_pipeline()

    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp, total_fp, total_fn = 0, 0, 0

    print(f"Evaluating NER on {len(GOLD)} gold-labeled descriptions...\n")

    for i, (text, annotations) in enumerate(GOLD, 1):
        gold_spans = resolve_gold_spans(text, annotations)
        pred_ents = [e for e in extract_entities(nlp, text)
                     if e["label"] in {"CVE_ID", "CWE_ID", "VERSION_RANGE", "VENDOR_PRODUCT", "ATTACK_VECTOR"}]

        tp, fp, fn = span_match(gold_spans, pred_ents)
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for (_, _, label) in tp:
            per_label[label]["tp"] += 1
        for (_, _, label) in fp:
            per_label[label]["fp"] += 1
        for (_, _, label) in fn:
            per_label[label]["fn"] += 1

        status = "OK" if not fp and not fn else "MISS"
        print(f"[{i:02d}] {status:4} tp={len(tp)} fp={len(fp)} fn={len(fn)}  \"{text[:70]}...\"")
        if fp:
            print(f"       false positives: {sorted(fp)}")
        if fn:
            print(f"       false negatives: {sorted(fn)}")

    print("\n" + "=" * 60)
    print("Per-entity-type results:")
    print(f"{'Label':<16}{'Precision':<12}{'Recall':<12}{'F1':<8}{'Support'}")
    for label in sorted(per_label):
        s = per_label[label]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"{label:<16}{p:<12.3f}{r:<12.3f}{f1:<8.3f}{s['tp'] + s['fn']}")

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    print("=" * 60)
    print(f"Overall micro-averaged: Precision={micro_p:.3f}  Recall={micro_r:.3f}  F1={micro_f1:.3f}")
    print(f"(TP={total_tp}, FP={total_fp}, FN={total_fn}, gold examples={len(GOLD)})")

    results = {
        "n_examples": len(GOLD),
        "per_label": {
            label: {
                "precision": round(s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0, 3),
                "recall": round(s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0, 3),
                "support": s["tp"] + s["fn"],
            } for label, s in per_label.items()
        },
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
    }
    out_path = Path(__file__).parent.parent.parent / "data" / "eval" / "ner_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

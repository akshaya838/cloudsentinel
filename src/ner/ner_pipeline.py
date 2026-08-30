"""
Named entity recognition for CVE / security text.

Off-the-shelf spaCy models aren't trained on security-domain entities (CVE IDs,
CWE IDs, version strings, attack techniques), so this pipeline layers a rule-based
EntityRuler on top of spaCy's base statistical NER model. This is a standard
low-resource-domain-adaptation approach (relevant to the Few-Sample NER literature
in the project's methodology) and gives good precision on structured entities
without needing a fine-tuned model yet.

Entity types produced:
    CVE_ID        - e.g. CVE-2021-44228
    CWE_ID        - e.g. CWE-502
    VERSION_RANGE - e.g. "2.0-beta9 through 2.14.1", "prior to 116.0.5845.187"
    VENDOR_PRODUCT- known vendor/product names (small seed dictionary, extend as needed)
    ATTACK_VECTOR - keyword-matched attack technique phrases
    ORG / PRODUCT - spaCy's generic statistical entities (fallback coverage)

Usage:
    python src/ner/ner_pipeline.py --in data/raw/sample_cves.json --out data/processed/cves_ner.json
"""

import argparse
import json
import re
from pathlib import Path

import spacy
from spacy.pipeline import EntityRuler

# Seed vendor/product dictionary — extend as the corpus grows.
# In a later iteration this should come from the CPE dictionary rather than
# being hand-maintained.
KNOWN_PRODUCTS = [
    "apache log4j2", "log4j", "spring framework", "apache tomcat", "google chrome",
    "winrar", "xz utils", "openssl", "microsoft windows", "linux kernel",
    "docker", "kubernetes", "jenkins", "nginx", "postgresql", "mysql",
    "wordpress", "gitlab", "citrix netscaler", "fortinet fortios", "vmware esxi",
]

ATTACK_VECTOR_KEYWORDS = [
    "remote code execution", "arbitrary code execution", "sql injection",
    "cross-site scripting", "buffer overflow", "heap buffer overflow",
    "denial of service", "privilege escalation", "path traversal",
    "deserialization", "authentication bypass", "man-in-the-middle",
    "server-side request forgery", "command injection",
]

CVE_ID_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
CWE_ID_PATTERN = re.compile(r"CWE-\d{1,4}", re.IGNORECASE)

# NOTE (found during Phase-I evaluation): version numbers can have more than
# 3 segments (e.g. Chrome's 116.0.5845.187), so the numeric part allows 1-3
# dot-separated groups after the first. Also, the "range" alternative (X
# through Y) captures its own span in the "range" named group so the leading
# "versions " qualifier word can be stripped from the reported entity text —
# earlier this word was inconsistently included, causing span drift against
# hand-labeled gold data. The "prior to"/"before" alternatives keep their
# qualifier word since it's meaningful there (there's no second bound to pair
# it with).
_VNUM = r"\d+(?:\.\d+){1,3}[a-z]?(?:-\w+)?"
VERSION_RANGE_PATTERN = re.compile(
    rf"(?:versions?\s+)?(?P<range>{_VNUM}\s*(?:through|to|-)\s*{_VNUM})"
    rf"|(?P<bound>prior to\s+{_VNUM}|before\s+{_VNUM})",
    re.IGNORECASE,
)


def build_pipeline() -> spacy.language.Language:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' not found. Run:\n"
            "    python -m spacy download en_core_web_sm"
        )

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []

    for product in KNOWN_PRODUCTS:
        patterns.append({"label": "VENDOR_PRODUCT", "pattern": [{"LOWER": t} for t in product.split()]})

    for phrase in ATTACK_VECTOR_KEYWORDS:
        patterns.append({"label": "ATTACK_VECTOR", "pattern": [{"LOWER": t} for t in phrase.split()]})

    ruler.add_patterns(patterns)
    return nlp


def regex_entities(text: str) -> list[dict]:
    entities = []
    for match in CVE_ID_PATTERN.finditer(text):
        entities.append({"text": match.group(), "label": "CVE_ID", "start": match.start(), "end": match.end()})
    for match in CWE_ID_PATTERN.finditer(text):
        entities.append({"text": match.group(), "label": "CWE_ID", "start": match.start(), "end": match.end()})
    for match in VERSION_RANGE_PATTERN.finditer(text):
        if match.group("range") is not None:
            start, end = match.start("range"), match.end("range")
        else:
            start, end = match.start("bound"), match.end("bound")
        entities.append({"text": text[start:end], "label": "VERSION_RANGE", "start": start, "end": end})
    return entities


def extract_entities(nlp: spacy.language.Language, text: str) -> list[dict]:
    doc = nlp(text)
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    entities.extend(regex_entities(text))

    # Drop duplicate/overlapping spans (regex entities take priority over generic spaCy ones)
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    deduped, last_end = [], -1
    for ent in entities:
        if ent["start"] >= last_end:
            deduped.append(ent)
            last_end = ent["end"]
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Run NER over CVE descriptions.")
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    args = parser.parse_args()

    nlp = build_pipeline()
    records = json.loads(Path(args.in_path).read_text())

    for record in records:
        record["entities"] = extract_entities(nlp, record.get("description", ""))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    print(f"Processed {len(records)} records -> {out_path}")

    # Quick sanity print for the first record
    if records:
        print("\nSample output:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

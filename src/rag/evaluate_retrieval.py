"""
Evaluate retrieval quality of the RAG index against a small query set.

Each query has a known correct source CVE (the record it should retrieve).
Metric: Hit-Rate@k — is the correct source_id present in the top-k results?
Also reports Mean Reciprocal Rank (MRR) for a finer-grained view.

Requires the index to already be built (see build_index.py) and needs
network access to huggingface.co on first run to download the
paraphrase-multilingual-MiniLM-L12-v2 model (cached locally after that).

Usage:
    python src/rag/build_index.py --in ../../data/processed/cves_classified.json \
        --collection cve_knowledge --persist-dir ../../data/processed/chroma_store
    python src/rag/evaluate_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import query_index

# (query, expected_source_cve_id)
QUERY_SET = [
    ("What vulnerability allowed Log4j to execute arbitrary code from LDAP servers?", "CVE-2021-44228"),
    ("Tell me about the Chrome WebP heap buffer overflow", "CVE-2023-4863"),
    ("What is Spring4Shell?", "CVE-2022-22965"),
    ("Explain the WinRAR archive extension spoofing vulnerability", "CVE-2023-38831"),
    ("What happened with the XZ Utils backdoor?", "CVE-2024-3094"),
    ("Describe the Heartbleed OpenSSL vulnerability", "CVE-2014-0160"),
    ("What CVE is associated with the Equifax breach?", "CVE-2017-5638"),
    ("What is EternalBlue and which CVE does it exploit?", "CVE-2017-0144"),
    ("Explain the Shellshock Bash vulnerability", "CVE-2014-6271"),
    ("What is PrintNightmare?", "CVE-2021-34527"),
    ("What SSRF vulnerability affected Microsoft Exchange, known as ProxyLogon?", "CVE-2021-26855"),
    ("What is the Follina MSDT vulnerability?", "CVE-2022-30190"),
    ("Describe the Dirty COW Linux kernel privilege escalation", "CVE-2016-5195"),
    ("What is BlueKeep?", "CVE-2019-0708"),
    ("Explain ZeroLogon", "CVE-2020-1472"),
    ("What SQL injection vulnerability affected MOVEit Transfer?", "CVE-2023-34362"),
    ("What is Dirty Pipe?", "CVE-2022-0847"),
    ("Explain the POODLE SSL 3.0 attack", "CVE-2014-3566"),
    # Harder queries: paraphrased, no CVE nickname or product name repeated verbatim —
    # tests semantic retrieval rather than keyword overlap.
    ("Which vulnerability let attackers run code on a Java logging library by controlling log message content?", "CVE-2021-44228"),
    ("A compression tool let attackers hide code execution behind a file with a mismatched extension", "CVE-2023-38831"),
    ("Which bug let a supply-chain attacker plant a hidden backdoor in a data-compression library used by SSH?", "CVE-2024-3094"),
    ("What image-decoding bug in a web browser corrupted heap memory via a malicious webpage?", "CVE-2023-4863"),
    ("Which Java web framework flaw was exploited through data binding when hosted on a servlet container?", "CVE-2022-22965"),
    ("What flaw in a Windows authentication protocol let an attacker impersonate a domain controller?", "CVE-2020-1472"),
]


def main():
    top_k = 3
    hits_at_1, hits_at_k, reciprocal_ranks = 0, 0, []

    print(f"Evaluating retrieval on {len(QUERY_SET)} queries (top-{top_k})...\n")

    for query, expected in QUERY_SET:
        results = query_index(query, collection_name="cve_knowledge",
                               persist_dir="../../data/processed/chroma_store", top_k=top_k)
        retrieved_ids = [r["metadata"]["source_id"] for r in results]

        rank = retrieved_ids.index(expected) + 1 if expected in retrieved_ids else None
        reciprocal_ranks.append(1 / rank if rank else 0)
        if rank == 1:
            hits_at_1 += 1
        if rank is not None:
            hits_at_k += 1

        status = f"rank {rank}" if rank else "MISS"
        print(f"[{status:>7}] \"{query}\" -> expected {expected}, got {retrieved_ids}")

    n = len(QUERY_SET)
    print("\n" + "=" * 60)
    print(f"Hit-Rate@1: {hits_at_1}/{n} = {hits_at_1/n:.3f}")
    print(f"Hit-Rate@{top_k}: {hits_at_k}/{n} = {hits_at_k/n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks)/n:.3f}")

    out_path = Path(__file__).parent.parent.parent / "data" / "eval" / "retrieval_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_queries": n, "hit_rate_at_1": round(hits_at_1/n, 3),
        f"hit_rate_at_{top_k}": round(hits_at_k/n, 3),
        "mrr": round(sum(reciprocal_ranks)/n, 3),
    }, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

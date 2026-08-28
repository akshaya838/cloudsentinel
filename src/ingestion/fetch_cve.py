"""
Fetch CVE records from the public NVD API (v2.0) and save them as JSON.

NVD API docs: https://nvd.nist.gov/developers/vulnerabilities
No API key is required, but requests are rate-limited (5 req/30s without a key,
50 req/30s with one). Set NVD_API_KEY as an environment variable if you have one.

Usage:
    python src/ingestion/fetch_cve.py --days 30 --severity HIGH --out data/raw/cves.json
    python src/ingestion/fetch_cve.py --keyword "apache log4j" --out data/raw/cves.json
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def build_params(args: argparse.Namespace) -> dict:
    params = {"resultsPerPage": args.results_per_page}

    if args.days:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        params["pubStartDate"] = start.strftime("%Y-%m-%dT00:00:00.000")
        params["pubEndDate"] = end.strftime("%Y-%m-%dT23:59:59.999")

    if args.severity:
        params["cvssV3Severity"] = args.severity.upper()

    if args.keyword:
        params["keywordSearch"] = args.keyword

    return params


def fetch_cves(params: dict, api_key: str | None) -> list[dict]:
    headers = {"apiKey": api_key} if api_key else {}
    all_results = []
    start_index = 0

    while True:
        query = dict(params, startIndex=start_index)
        resp = requests.get(NVD_BASE_URL, params=query, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        vulns = data.get("vulnerabilities", [])
        all_results.extend(v["cve"] for v in vulns)

        total = data.get("totalResults", 0)
        start_index += len(vulns)

        print(f"Fetched {start_index}/{total} CVEs")

        if start_index >= total or not vulns:
            break

        # Respect NVD rate limits
        time.sleep(6 if not api_key else 0.6)

    return all_results


def simplify(cve: dict) -> dict:
    """Pull out the fields we actually need for downstream NER/classification."""
    description = next(
        (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), ""
    )

    metrics = cve.get("metrics", {})
    cvss = None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            cvss = metrics[key][0]["cvssData"]
            break

    cwe_ids = [
        d["value"]
        for w in cve.get("weaknesses", [])
        for d in w.get("description", [])
        if d["value"].startswith("CWE-")
    ]

    return {
        "cve_id": cve["id"],
        "description": description,
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "cvss_base_score": cvss.get("baseScore") if cvss else None,
        "cvss_severity": cvss.get("baseSeverity") if cvss else None,
        "cwe_ids": cwe_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch CVE data from NVD.")
    parser.add_argument("--days", type=int, default=30, help="Look back N days by publish date")
    parser.add_argument("--severity", type=str, default=None, help="LOW / MEDIUM / HIGH / CRITICAL")
    parser.add_argument("--keyword", type=str, default=None, help="Keyword search (e.g. product name)")
    parser.add_argument("--results-per-page", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/raw/cves.json")
    args = parser.parse_args()

    api_key = os.environ.get("NVD_API_KEY")
    params = build_params(args)

    print(f"Querying NVD with params: {params}")
    raw = fetch_cves(params, api_key)
    simplified = [simplify(c) for c in raw]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(simplified, indent=2, ensure_ascii=False))

    print(f"Saved {len(simplified)} CVEs to {out_path}")


if __name__ == "__main__":
    main()

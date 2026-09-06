"""
Fetch device adverse event reports (MAUDE) from openFDA into a local cache.

openFDA data is public domain under CC0 1.0: redistribution and commercial use
are permitted, with no attribution legally required (we attribute anyway). The
one carve-out is GMDN device terminology, which requires separate licensing --
this fetcher does not request or store GMDN fields.

WHY THIS IS A SEPARATE SCRIPT
    Fetching happens once and writes a cache. Everything downstream reads the
    cache and makes no network calls, so results reproduce offline and byte for
    byte. Run this on a machine with open network access to api.fda.gov.

    python -m src.fetch_maude --n 20000 --out data/maude_raw.jsonl

Notes on the API
    limit  max 1000 per call
    skip   max 25000, so a single query reaches ~26k records; for more, narrow
           the query by date range and concatenate (--split-years does this).
    A free API key raises the daily quota substantially and is read from the
    OPENFDA_API_KEY environment variable if present.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.fda.gov/device/event.json"
MAX_LIMIT = 1000
MAX_SKIP = 25000

# Reports carrying both a narrative and a recorded remedial action are the ones
# this study can use; everything else is dropped at build time anyway.
DEFAULT_QUERY = (
    "_exists_:remedial_action AND _exists_:product_problems "
    "AND _exists_:device.device_report_product_code"
)


# A browser-like User-Agent. Some CDN edges in front of public APIs reject
# unfamiliar agents with 403 before the query is ever parsed.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class Refused(RuntimeError):
    """openFDA refused the request for a reason retrying will not fix."""


def _get(url: str, timeout: int = 60, retries: int = 4) -> dict:
    headers = {"User-Agent": UA, "Accept": "application/json"}
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2          # backoff, then try again
                continue
            if e.code == 403:
                raise Refused(
                    "openFDA returned 403 Forbidden.\n"
                    "  This is a refusal, not a bad query. Usual causes:\n"
                    "    1. Daily quota exhausted for this IP. Without an API key\n"
                    "       openFDA allows ~1,000 requests/day per IP, and a shared\n"
                    "       corporate address can burn that without you.\n"
                    "       Fix: get a free key at\n"
                    "       https://open.fda.gov/apis/authentication/\n"
                    "       then:  $env:OPENFDA_API_KEY = \"your-key\"\n"
                    "    2. A proxy or network filter between you and api.fda.gov.\n"
                    "       Test:  (Invoke-WebRequest \"https://api.fda.gov/device/\"\n"
                    "              + \"event.json?limit=1\").StatusCode\n"
                    "    3. User-Agent rejection. This client now sends a browser\n"
                    "       agent, so this is the least likely of the three.") from e
            raise
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise Refused(
                f"could not reach {ENDPOINT}: {e.reason}\n"
                "  If you are behind a corporate proxy, set HTTPS_PROXY and retry."
            ) from e
    raise Refused("exhausted retries")


def _page(query: str, limit: int, skip: int) -> list[dict]:
    params: dict = {"limit": limit, "skip": skip}
    if query:                      # empty query -> no search parameter at all
        params["search"] = query
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        params["api_key"] = key
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    try:
        return _get(url).get("results", [])
    except urllib.error.HTTPError as e:
        if e.code == 404:          # openFDA returns 404 when a page is empty
            return []
        raise


def fetch(n: int, query: str = DEFAULT_QUERY, pause: float = 0.3,
          verbose: bool = True, page_size: int = MAX_LIMIT) -> list[dict]:
    out: list[dict] = []
    skip = 0
    while len(out) < n and skip <= MAX_SKIP:
        batch = _page(query, min(page_size, n - len(out)), skip)
        if not batch:
            break
        out.extend(batch)
        skip += len(batch)
        if verbose:
            print(f"  fetched {len(out):>6} / {n}", end="\r", flush=True)
        time.sleep(pause)
    if verbose:
        print()
    return out


def fetch_by_year(n_per_year: int, years: list[int], query: str = DEFAULT_QUERY,
                  verbose: bool = True, page_size: int = MAX_LIMIT) -> list[dict]:
    """
    Split by receipt year to get past the 25k skip ceiling, and to give the
    corpus a defined time span, which the recurrence derivation needs.
    """
    out: list[dict] = []
    for y in years:
        q = (f"{query} AND date_received:[{y}0101 TO {y}1231]" if query
             else f"date_received:[{y}0101 TO {y}1231]")
        if verbose:
            print(f"  {y}:")
        got = fetch(n_per_year, q, verbose=verbose, page_size=page_size)
        if verbose:
            print(f"  {y}: {len(got)} reports")
        out.extend(got)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000,
                    help="reports per year when --split-years, else total")
    ap.add_argument("--years", nargs="*", type=int,
                    default=list(range(2018, 2025)))
    ap.add_argument("--split-years", action="store_true", default=True)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--no-search", action="store_true",
                    help="fetch without a search filter and filter locally; use "
                         "this if a firewall rejects the query syntax")
    ap.add_argument("--page-size", type=int, default=MAX_LIMIT,
                    help="records per request (lower it if large pages are refused)")
    ap.add_argument("--out", default="data/maude_raw.jsonl")
    a = ap.parse_args()

    print("openFDA device/event (MAUDE) -- public domain, CC0 1.0")
    if os.environ.get("OPENFDA_API_KEY"):
        print("using OPENFDA_API_KEY from the environment")
    else:
        print("no OPENFDA_API_KEY set - keyless quota is ~1,000 requests/day per IP.")
        print("A free key takes a minute: https://open.fda.gov/apis/authentication/")
    try:
        query = "" if a.no_search else a.query
        rows = (fetch_by_year(a.n, a.years, query, page_size=a.page_size)
                if a.split_years else fetch(a.n, query, page_size=a.page_size))
    except Refused as e:
        raise SystemExit(f"\n{e}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "source": "openFDA device/event (MAUDE)",
        "endpoint": ENDPOINT,
        "licence": "CC0 1.0 Universal (public domain dedication)",
        "licence_url": "https://open.fda.gov/license/",
        "query": a.query,
        "years": a.years,
        "n_records": len(rows),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "GMDN terminology fields are not requested or stored.",
    }
    Path(out.parent / "maude_provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote {len(rows)} raw reports -> {out}")
    print(f"provenance -> {out.parent / 'maude_provenance.json'}")


LADDER = [
    ("plain, limit=1",            {"limit": 1}),
    ("plain, limit=1000",         {"limit": 1000}),
    ("simple search",             {"search": "event_type:Malfunction", "limit": 1}),
    ("date range",                {"search": "date_received:[20180101 TO 20181231]",
                                   "limit": 1}),
    ("_exists_ single",           {"search": "_exists_:remedial_action", "limit": 1}),
    ("_exists_ + AND",            {"search": "_exists_:remedial_action AND "
                                             "_exists_:product_problems", "limit": 1}),
    ("full default query",        {"search": DEFAULT_QUERY, "limit": 1}),
    ("full query + date + 1000",  {"search": DEFAULT_QUERY +
                                   " AND date_received:[20180101 TO 20181231]",
                                   "limit": 1000}),
]


def diagnose() -> None:
    """
    Climb from the simplest possible request to the one the fetcher actually
    makes, and report the first rung that fails. That identifies whether the
    problem is the endpoint, the search syntax, the page size, or the URL.

        python -m src.fetch_maude --diagnose
    """
    key = os.environ.get("OPENFDA_API_KEY")
    print(f"endpoint {ENDPOINT}")
    print(f"api_key  {'set' if key else 'not set'}\n")
    first_fail = None
    for label, params in LADDER:
        if key:
            params = {**params, "api_key": key}
        url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
        try:
            data = _get(url, retries=1)
            total = data.get("meta", {}).get("results", {}).get("total")
            got = len(data.get("results", []))
            print(f"  OK    {label:<26} returned {got:>4}"
                  f"{f'  (total {total:,} match)' if total else ''}")
        except urllib.error.HTTPError as e:
            print(f"  {e.code}   {label:<26} {e.reason}")
            first_fail = first_fail or (label, e.code, url)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {label:<26} {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:60]}")
            first_fail = first_fail or (label, "err", url)

    print()
    if not first_fail:
        print("  Every rung passed. Re-run `python run.py fetch`.")
        return
    label, code, url = first_fail
    print(f"  First failure: {label}  ({code})")
    print(f"  URL: {url[:150]}")
    print()
    if label in ("plain, limit=1", "plain, limit=1000"):
        print("  The endpoint itself is refused -> network, proxy or quota.")
    elif "limit=1000" in label:
        print("  Only the large page size fails -> retry with --page-size 100.")
    else:
        print("  A search query is refused while plain requests succeed.")
        print("  That is the signature of a web application firewall reacting to")
        print("  the query syntax (_exists_, AND, brackets), not of openFDA.")
        print("  Workaround: fetch without a search filter and filter locally --")
        print("      python run.py fetch --no-search")


def probe() -> None:
    """One request, maximum diagnostics. `python -m src.fetch_maude --probe`."""
    url = f"{ENDPOINT}?limit=1"
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        url += f"&api_key={key}"
    print(f"GET {ENDPOINT}?limit=1"
          f"{'  (with api_key)' if key else '  (no api_key)'}")
    try:
        data = _get(url, retries=1)
        n = data.get("meta", {}).get("results", {}).get("total")
        print(f"  OK - endpoint reachable, {n:,} total records match an empty query"
              if n else "  OK - endpoint reachable")
        print("  So the 403 was not the network. Re-run the fetch.")
    except Refused as e:
        print(f"  REFUSED\n{e}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import sys as _sys
    if "--diagnose" in _sys.argv:
        diagnose()
    elif "--probe" in _sys.argv:
        probe()
    else:
        main()

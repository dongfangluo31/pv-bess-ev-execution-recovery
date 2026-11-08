import argparse
import base64
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request

BASE = "https://ev.caltech.edu/api/v1/"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(site, output, token):
    endpoint = urljoin(BASE, f"sessions/{site}")
    expected = urlsplit(endpoint)
    current = endpoint + "?max_results=100&sort=connectionTime"
    opener = build_opener(NoRedirect())
    header = "Basic " + base64.b64encode((token + ":").encode("utf-8")).decode("ascii")
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    records = 0
    total = None
    while current is not None:
        location = urlsplit(current)
        if (location.scheme, location.netloc, location.path) != (expected.scheme, expected.netloc, expected.path):
            raise ValueError("Unexpected ACN pagination target")
        if current in seen:
            raise ValueError("ACN returned a repeated page")
        seen.add(current)
        request = Request(current, headers={"Authorization":header, "Accept":"application/json"})
        with opener.open(request, timeout=60) as response:
            body = response.read()
        page = json.loads(body)
        number = len(seen)
        meta = page["_meta"]
        if meta["page"] != number:
            raise ValueError("Unexpected ACN page number")
        if total is None:
            total = meta["total"]
        if total != meta["total"]:
            raise ValueError("ACN population changed during download")
        with (output/f"page_{number:06d}.json").open("xb") as fh:
            fh.write(body)
        records += len(page["_items"])
        next_link = page.get("_links",{}).get("next",{}).get("href")
        if records>total or (next_link and (not page["_items"] or records>=total)):
            raise ValueError("Invalid ACN page population")
        current = urljoin(BASE, next_link) if next_link else None
    if records != total:
        raise ValueError(f"Incomplete download: {records} of {total} sessions")
    print(json.dumps({"site":site,"pages":len(seen),"sessions":records}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["caltech","jpl","office001"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directory = args.out.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if directory == checkout or checkout in directory.parents:
        parser.error("Choose an empty new directory outside the public repository for individual ACN records")
    token = os.environ.get("ACN_API_TOKEN")
    if not token:
        parser.error("Set ACN_API_TOKEN to your own ACN-Data access token")
    fetch(args.site, directory, token)


if __name__=="__main__":
    main()

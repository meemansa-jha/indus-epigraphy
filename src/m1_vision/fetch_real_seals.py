"""Download freely-licensed Indus seal images from Wikimedia Commons (thumbnails).

Commons images are openly licensed (CC / public domain). This pulls files from a
Commons category into data/raw/real_seals/ as ~1024px THUMBNAILS (full-size
originals are heavily rate-limited; thumbnails are plenty for sign recognition),
and records licence + attribution for each. Standard library only.

Usage:
  python src/m1_vision/fetch_real_seals.py --limit 150
  python src/m1_vision/fetch_real_seals.py --category "Category:Harappan seals" --width 1200
"""
from __future__ import annotations
import argparse, csv, json, re, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path

from config import RAW_DIR

API = "https://commons.wikimedia.org/w/api.php"
UA = "indus-epigraphy-coursework/1.0 (educational research; non-commercial)"
OUT = RAW_DIR / "real_seals"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def http_json(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 5 * (i + 1)
                print(f"    rate-limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return {}


def api_get(params):
    return http_json(API + "?" + urllib.parse.urlencode({**params, "format": "json"}))


def list_category_files(category, limit):
    got, cont = 0, {}
    while got < limit:
        data = api_get({"action": "query", "list": "categorymembers",
                        "cmtitle": category, "cmtype": "file",
                        "cmlimit": min(500, limit - got), **cont})
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]; got += 1
        if "continue" in data:
            cont = data["continue"]; time.sleep(0.5)
        else:
            break


def image_info(title, width):
    data = api_get({"action": "query", "titles": title, "prop": "imageinfo",
                    "iiprop": "url|extmetadata", "iiurlwidth": width})
    for _, page in data.get("query", {}).get("pages", {}).items():
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        return {"url": info.get("thumburl") or info.get("url"),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "artist": meta.get("Artist", {}).get("value", ""),
                "credit": meta.get("Credit", {}).get("value", "")}
    return None


def download(url, dest, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 5 * (i + 1)
                print(f"    rate-limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return False


def clean(s):
    return re.sub("<[^>]+>", "", s or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Category:Indus civilization seals")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--width", type=int, default=1024, help="thumbnail width in px")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    print(f"Listing files in {args.category!r} (thumbnails @ {args.width}px) ...")
    for title in list_category_files(args.category, args.limit):
        if not title.lower().endswith(IMG_EXTS):
            continue
        try:
            info = image_info(title, args.width)
            if not info or not info["url"]:
                continue
            fname = title.replace("File:", "").replace(" ", "_")
            dest = OUT / fname
            if not dest.exists():
                download(info["url"], dest)
                time.sleep(1.0)                 # polite pacing
            rows.append({"file": fname, "license": clean(info["license"]),
                         "artist": clean(info["artist"]), "credit": clean(info["credit"]),
                         "source_url": info["url"]})
            n += 1
            print(f"  [{n}] {fname}  ({clean(info['license'])})")
        except Exception as e:
            print(f"  skip {title}: {e}")

    with open(OUT / "attribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "license", "artist", "credit", "source_url"])
        w.writeheader(); w.writerows(rows)
    print(f"\nDone. {n} images -> {OUT}")
    print(f"Attribution saved -> {OUT/'attribution.csv'}")


if __name__ == "__main__":
    main()

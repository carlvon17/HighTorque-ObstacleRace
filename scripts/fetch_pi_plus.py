"""Fetch pinned upstream assets; verify Git blob hashes before using them."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    target = ROOT / "vendor/mini_pi_plus_amp"

    def fetch(entry):
        path = target / entry["path"]
        def valid(data):
            blob = f"blob {len(data)}\0".encode() + data
            return hashlib.sha1(blob).hexdigest() == entry["sha"]
        if path.exists() and valid(path.read_bytes()):
            return
        url = f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/{entry['path']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if not valid(data):
            raise ValueError(f"Checksum mismatch: {entry['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"Fetched {entry['path']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, lock["files"]))
    (target / "SOURCE.json").write_text(json.dumps(lock, indent=2))
    print(f"Verified {len(lock['files'])} files at {lock['commit']}")


if __name__ == "__main__":
    main()

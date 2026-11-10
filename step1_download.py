"""Step 1: download the seven public EIA-930 BALANCE six-month files into raw/ and print each file's size and SHA-256.

Files already present with the size the server reports are kept. Writes out/step1_files.csv.
The newest file (2026_Jul_Dec) is extended by EIA every day, so its checksum changes between runs.
"""
import csv, hashlib, os, sys, time, urllib.request
from datetime import datetime, timezone
from common import *

UA = {'User-Agent': 'runkit-eia930/1.0 (research rerun)'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method='HEAD'), timeout=60) as r:
        return int(r.headers['Content-Length'])


def fetch(url, dest):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest + '.part', 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(dest + '.part', dest)
            return
        except Exception as e:
            print(f'  retry {attempt + 1}: {e}')
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f'Could not download {url}')


rows = []
print(f'{"file":34s} {"MB":>8s}  sha256')
for tag in BAL_FILES:
    name = f'EIA930_BALANCE_{tag}.csv'
    url, dest = EIA_BASE + name, bal_path(tag)
    if not (os.path.exists(dest) and os.path.getsize(dest) == remote_size(url)):
        fetch(url, dest)
    digest = sha256(dest)
    rows.append(dict(file=name, url=url, bytes=os.path.getsize(dest), sha256=digest, checked_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')))
    print(f'{name:34s} {os.path.getsize(dest) / 1e6:8.1f}  {digest}', flush=True)

with open(os.path.join(OUT, 'step1_files.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
total = sum(r['bytes'] for r in rows)
print(f'\n{len(rows)} files, {total / 1e6:.0f} MB downloaded to raw/. Record written to out/step1_files.csv.')
print('Meaning: these are the exact public files every later step reads. Compare the checksums with a second run or a colleague to show the inputs are identical.')

#!/usr/bin/env bash
# Fail when a submission zip is over 1 GB (10^9 bytes).
# Prints the zip size and the largest entries.
#
#   scripts/check_zip_budget.sh submission.zip
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/check_zip_budget.sh path/to.zip" >&2
  exit 2
fi

ZIP="$1"
if [[ ! -f "$ZIP" ]]; then
  echo "missing zip: $ZIP" >&2
  exit 2
fi

LIMIT=1000000000
bytes="$(stat -c%s "$ZIP")"

python3 - "$ZIP" "$bytes" "$LIMIT" <<'PY'
import sys
import zipfile

zip_path, bytes_s, limit_s = sys.argv[1:]
size = int(bytes_s)
limit = int(limit_s)
print(f"zip: {zip_path}")
print(f"size_bytes: {size}")
print(f"size_MiB: {size / 1024 / 1024:.2f}")
print(f"limit_bytes: {limit} (1 GB = 10^9 bytes)")
with zipfile.ZipFile(zip_path) as archive:
    infos = sorted(archive.infolist(), key=lambda item: item.file_size, reverse=True)
    print(f"entries: {len(infos)}")
    for info in infos[:40]:
        print(f"  {info.file_size:12d}  {info.filename}")
    if len(infos) > 40:
        print(f"  ... {len(infos) - 40} more")
    names = [info.filename for info in infos]
    banned = [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
    if banned:
        print("FAIL: zip contains bytecode:", ", ".join(banned), file=sys.stderr)
        sys.exit(1)
if size > limit:
    print(
        f"FAIL: zip exceeds 1 GB by {size - limit} bytes",
        file=sys.stderr,
    )
    sys.exit(1)
print("OK: zip is within 1 GB")
PY

import hashlib
import re
from pathlib import Path

content = Path("SATQUERY-P4-PROTOCOL-V1.12.1-DRAFT.md").read_text(encoding="utf-8")

m = re.search(r"### Complete Cryptographic Provenance Manifest.*?(?=### Empirical Reproducibility Policy)", content, re.DOTALL)
lines = [l.strip() for l in m.group(0).splitlines() if l.strip().startswith("| **")]

print(f"Total manifest table entries: {len(lines)}\n")

table_rows = []
mismatches = 0

for idx, line in enumerate(lines, 1):
    cols = [c.strip() for c in line.split("|")[1:-1]]
    name = cols[0].replace("**", "")
    
    # Extract backtick contents
    path_match = re.search(r"`([^`]+)`", cols[1])
    path_str = path_match.group(1) if path_match else cols[1]

    hash_match = re.search(r"`([^`]+)`", cols[2])
    manifest_hash = hash_match.group(1) if hash_match else cols[2]

    category = cols[3]

    p = Path(path_str)
    if not p.exists():
        status = "MISSING"
        disk_hash = "N/A"
        mismatches += 1
    else:
        disk_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if disk_hash.lower() == manifest_hash.lower():
            status = "VERIFIED MATCH"
        else:
            status = "MISMATCH"
            mismatches += 1

    table_rows.append({
        "index": idx,
        "name": name,
        "path": path_str,
        "manifest_hash": manifest_hash,
        "disk_hash": disk_hash,
        "status": status,
        "category": category
    })
    print(f"{idx:02d}. [{status}] {path_str}")
    print(f"    Manifest: {manifest_hash}")
    print(f"    Disk:     {disk_hash}")

print(f"\nAudit Result: {len(lines) - mismatches}/{len(lines)} verified. Total discrepancies: {mismatches}")

import hashlib
import re
from pathlib import Path

content = Path("SATQUERY-P4-PROTOCOL-V1.12.1-DRAFT.md").read_text(encoding="utf-8")

m = re.search(r"### Complete Cryptographic Provenance Manifest.*?(?=### Empirical Reproducibility Policy)", content, re.DOTALL)
lines = [l.strip() for l in m.group(0).splitlines() if l.strip().startswith("| **")]

print(f"Total manifest table entries: {len(lines)}\n")

table_rows = []
mismatches = 0

for line in lines:
    cols = [c.strip() for c in line.split("|")[1:-1]]
    name = cols[0].replace("**", "")
    # extract path inside backticks
    path_match = re.search(r"`([^`]+)`", cols[1])
    path_str = path_match.group(1) if path_match else cols[1]

    # extract hash inside backticks
    hash_match = re.search(r"`([^`]+)`", cols[2])
    expected_hash = hash_match.group(1) if hash_match else cols[2]

    category = cols[3]

    p = Path(path_str)
    if not p.exists():
        status = "MISSING FILE"
        actual_hash = "N/A"
        mismatches += 1
    else:
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_hash.lower() == expected_hash.lower():
            status = "VERIFIED MATCH"
        else:
            status = "MISMATCH"
            mismatches += 1

    table_rows.append((name, path_str, expected_hash, actual_hash, status))
    print(f"[{status}] {path_str} -> {actual_hash[:16]}...")

print(f"\nAudit Result: {len(lines) - mismatches}/{len(lines)} verified. Total discrepancies: {mismatches}")

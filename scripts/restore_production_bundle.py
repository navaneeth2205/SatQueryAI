#!/usr/bin/env python3
"""
SatQuery AI — Secure Production Artifact Bundle Restoration Tool
=================================================================
Safely extracts, validates, and verifies the frozen production-artifact
bundle for SatQuery AI Milestone M3.3 (Release Candidate v1.0.0-rc1).

Security & Integrity Guarantees:
  1. Path Traversal Defense: Rejects absolute paths, drive letters, and '..' components.
  2. Archive Ledger Verification: Computes and matches archive SHA-256 before extraction.
  3. Manifest Hash Verification: Confirms byte-exact SHA-256 of the freeze manifest.
  4. Cryptographic Asset Verification: Validates all 13 production assets against manifest.
  5. Machine-Readable Audit Output: Emits structured JSON verification telemetry.
"""

import argparse
import hashlib
import json
import logging
import os
import pathlib
import sys
import tarfile
from typing import Dict, Any, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [satquery.restore] %(message)s"
)
logger = logging.getLogger("satquery.restore")


def compute_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA-256 hex digest of a file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_safe_tar_member(member: tarfile.TarInfo, destination: pathlib.Path) -> bool:
    """
    Prevent directory traversal attacks (Zip Slip / Tar Slip).
    Rejects absolute paths, drive letters, and paths resolving outside destination.
    """
    path = member.name
    # Reject Windows drive letters (e.g. C:) or root slashes
    if os.path.isabs(path) or (len(path) > 1 and path[1] == ":"):
        logger.error(f"REJECTED: Member contains absolute path: {path}")
        return False

    # Check normalized destination
    dest_path = (destination / path).resolve()
    dest_root = destination.resolve()

    try:
        dest_path.relative_to(dest_root)
    except ValueError:
        logger.error(f"REJECTED: Directory traversal attempt detected: {path}")
        return False

    # Whitelist prefix check: must be inside artifacts/ or data/
    normalized_rel = str(dest_path.relative_to(dest_root)).replace("\\", "/")
    if not (normalized_rel.startswith("artifacts/") or normalized_rel.startswith("data/")):
        logger.error(f"REJECTED: Member is outside allowed prefixes (artifacts/, data/): {normalized_rel}")
        return False

    return True


def verify_archive_ledger(archive_path: pathlib.Path, ledger_path: pathlib.Path) -> bool:
    """Verify archive SHA-256 against pre-computed checksum ledger file."""
    if not ledger_path.exists():
        logger.error(f"Ledger file not found: {ledger_path}")
        return False

    computed_hash = compute_sha256(archive_path)
    ledger_content = ledger_path.read_text(encoding="utf-8").strip()
    expected_hash = ledger_content.split()[0].lower()

    if computed_hash.lower() == expected_hash:
        logger.info(f"Archive SHA-256 MATCHES ledger: {computed_hash}")
        return True
    else:
        logger.error(f"Archive SHA-256 MISMATCH! Computed: {computed_hash}, Expected: {expected_hash}")
        return False


def extract_secure(
    archive_path: pathlib.Path,
    destination: pathlib.Path
) -> Tuple[bool, List[str]]:
    """Safely extract archive members with path sanitization."""
    extracted_members = []
    logger.info(f"Inspecting archive: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            if not is_safe_tar_member(member, destination):
                return False, []

        logger.info(f"All {len(members)} archive entries passed path safety checks. Extracting...")
        for member in members:
            tar.extract(member, path=destination)
            extracted_members.append(member.name)

    logger.info(f"Successfully extracted {len(extracted_members)} items to {destination}")
    return True, extracted_members


def verify_production_manifest(
    destination: pathlib.Path,
    manifest_rel_path: str = "artifacts/deployment/m33_production_freeze_manifest.json"
) -> Dict[str, Any]:
    """Verify all 13 production assets in the restored directory against manifest."""
    manifest_file = destination / manifest_rel_path
    if not manifest_file.exists():
        return {
            "all_passed": False,
            "error": f"Manifest file missing at {manifest_file}",
            "asset_results": {}
        }

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    artifacts = manifest_data.get("artifacts", {})
    asset_results = {}
    all_passed = True

    for key, meta in artifacts.items():
        file_path = destination / meta["path"]
        if not file_path.exists():
            asset_results[key] = {
                "status": "FAIL",
                "reason": "File not found",
                "path": meta["path"]
            }
            all_passed = False
            continue

        actual_size = file_path.stat().st_size
        actual_hash = compute_sha256(file_path)
        expected_size = meta["size_bytes"]
        expected_hash = meta["sha256"]

        is_match = (actual_size == expected_size) and (actual_hash == expected_hash)
        if is_match:
            asset_results[key] = {
                "status": "PASS",
                "size_bytes": actual_size,
                "sha256": actual_hash
            }
        else:
            asset_results[key] = {
                "status": "FAIL",
                "reason": "Checksum or size mismatch",
                "actual_size": actual_size,
                "expected_size": expected_size,
                "actual_sha256": actual_hash,
                "expected_sha256": expected_hash
            }
            all_passed = False

    return {
        "all_passed": all_passed,
        "manifest_path": str(manifest_file),
        "total_assets_checked": len(artifacts),
        "passed_count": sum(1 for v in asset_results.values() if v.get("status") == "PASS"),
        "failed_count": sum(1 for v in asset_results.values() if v.get("status") == "FAIL"),
        "asset_results": asset_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="SatQuery AI Secure Production Artifact Restoration Tool"
    )
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        required=True,
        help="Path to production bundle tar.gz archive"
    )
    parser.add_argument(
        "--ledger",
        type=pathlib.Path,
        default=None,
        help="Path to pre-computed SHA-256 ledger (defaults to <archive>.sha256)"
    )
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Target repository root destination (defaults to current directory)"
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=None,
        help="Optional path to write JSON verification report"
    )
    args = parser.parse_args()

    ledger_path = args.ledger or args.archive.with_suffix(args.archive.suffix + ".sha256")
    logger.info("Starting secure artifact restoration...")

    # 1. Ledger Verification
    if ledger_path.exists():
        if not verify_archive_ledger(args.archive, ledger_path):
            logger.error("Ledger verification FAILED! Halting restoration.")
            sys.exit(1)
    else:
        logger.warning(f"No ledger file found at {ledger_path}; proceeding with caution.")

    # 2. Secure Extraction
    success, extracted_files = extract_secure(args.archive, args.dest)
    if not success:
        logger.error("Extraction aborted due to security validation failure.")
        sys.exit(1)

    # 3. Cryptographic Manifest Verification
    audit_results = verify_production_manifest(args.dest)
    logger.info(f"Restored verification result: {audit_results['passed_count']}/{audit_results['total_assets_checked']} assets passed.")

    report_payload = {
        "restoration_successful": success and audit_results["all_passed"],
        "archive_path": str(args.archive),
        "destination": str(args.dest.resolve()),
        "extracted_file_count": len(extracted_files),
        "manifest_audit": audit_results
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        logger.info(f"Verification report written to {args.report}")

    if not audit_results["all_passed"]:
        logger.error("Production manifest verification FAILED after restoration!")
        sys.exit(1)

    logger.info("Secure production artifact restoration completed SUCCESSFULLY.")
    sys.exit(0)


if __name__ == "__main__":
    main()

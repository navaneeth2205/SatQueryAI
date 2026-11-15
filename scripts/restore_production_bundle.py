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


ALLOWED_ROOT_FILES = {"requirements.txt", "pyproject.toml"}
ALLOWED_PREFIXES = ("artifacts/", "data/")
FROZEN_MANIFEST_SHA256 = "4414e6e31bd756f5804fb742ecfb0bc7dfd3ccdbe57009d0e3d1904c49e21831"


def compute_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA-256 hex digest of a file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_safe_tar_member(member: tarfile.TarInfo, destination: pathlib.Path) -> bool:
    """
    Comprehensive security validation of archive members.
    Rejects:
      - Symlinks & Hardlinks (prevent symlink overwrite/escape)
      - Special device nodes / FIFOs
      - Absolute paths (Unix '/' or Windows drive letters 'C:')
      - UNC network paths ('\\\\server\\share' or '//server/share')
      - Directory traversal sequences ('..')
      - Resolution outside destination root
      - Paths outside the whitelist: 'artifacts/', 'data/', or root manifests.
    """
    # 1. Type check: Only allow regular files and directories
    if member.issym() or member.islnk():
        logger.error(f"REJECTED: Member is a symlink or hardlink: {member.name} -> {member.linkname}")
        return False
    if member.ischr() or member.isblk() or member.isfifo():
        logger.error(f"REJECTED: Member is a special device node or FIFO: {member.name}")
        return False
    if not (member.isreg() or member.isdir()):
        logger.error(f"REJECTED: Member is not a regular file: {member.name}")
        return False

    # 2. Path sanitization
    raw_name = member.name.replace("\\", "/")

    # Check for UNC network paths
    if raw_name.startswith("//") or member.name.startswith("\\\\"):
        logger.error(f"REJECTED: UNC network path detected: {member.name}")
        return False

    # Check for Windows drive letters (e.g. C:, D:)
    if ":" in raw_name:
        logger.error(f"REJECTED: Windows drive letter detected: {member.name}")
        return False

    # Check for absolute paths
    if raw_name.startswith("/") or os.path.isabs(member.name):
        logger.error(f"REJECTED: Absolute path detected: {member.name}")
        return False

    # Check for path traversal segments ('..')
    path_parts = pathlib.PurePosixPath(raw_name).parts
    if ".." in path_parts:
        logger.error(f"REJECTED: Directory traversal '..' detected: {member.name}")
        return False

    # 3. Normalized destination boundary check
    dest_path = (destination / raw_name).resolve()
    dest_root = destination.resolve()

    try:
        rel_path = dest_path.relative_to(dest_root)
    except ValueError:
        logger.error(f"REJECTED: Path resolves outside destination boundary: {member.name}")
        return False

    # 4. Strict Whitelist Prefix & Root File Check
    normalized_rel = str(rel_path).replace("\\", "/")
    if member.isdir():
        if not (normalized_rel.startswith(ALLOWED_PREFIXES) or normalized_rel == "."):
            logger.error(f"REJECTED: Directory member outside allowed prefixes: {normalized_rel}")
            return False
    else:
        is_allowed_root = normalized_rel in ALLOWED_ROOT_FILES
        is_allowed_prefix = any(normalized_rel.startswith(p) for p in ALLOWED_PREFIXES)
        if not (is_allowed_root or is_allowed_prefix):
            logger.error(f"REJECTED: Member outside allowed whitelist (artifacts/, data/, root configs): {normalized_rel}")
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
    """Safely extract archive members with path sanitization and collision prevention."""
    extracted_members = []
    seen_members = set()
    logger.info(f"Inspecting archive: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            norm_name = str(pathlib.PurePosixPath(member.name.replace("\\", "/")))
            if norm_name in seen_members:
                logger.error(f"REJECTED: Duplicate member detected in archive: {member.name}")
                return False, []
            seen_members.add(norm_name)

            if not is_safe_tar_member(member, destination):
                return False, []

        logger.info(f"All {len(members)} archive entries passed path safety checks. Extracting...")
        for member in members:
            dest_file = destination / member.name
            if dest_file.is_file() and member.isreg():
                extracted_f = tar.extractfile(member)
                if extracted_f:
                    archive_bytes = extracted_f.read()
                    try:
                        disk_bytes = dest_file.read_bytes()
                        if archive_bytes == disk_bytes:
                            # Content is already bitwise identical on disk; preserve existing file and attributes
                            extracted_members.append(member.name)
                            continue
                    except Exception:
                        pass
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

    manifest_hash = compute_sha256(manifest_file)
    manifest_hash_match = (manifest_hash == FROZEN_MANIFEST_SHA256)
    if not manifest_hash_match:
        logger.warning(
            f"Manifest SHA-256 differs from frozen reference: computed {manifest_hash}, expected {FROZEN_MANIFEST_SHA256}"
        )

    return {
        "all_passed": all_passed and manifest_hash_match,
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_hash,
        "manifest_sha256_match": manifest_hash_match,
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
        "--expected-sha256",
        type=str,
        default=None,
        help="Trusted SHA-256 digest of the archive (independent verification anchor)"
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

    # 1. Archive Hash Verification (Trusted Pinned Anchor and/or Ledger)
    computed_archive_hash = compute_sha256(args.archive)
    if args.expected_sha256:
        if computed_archive_hash.lower() != args.expected_sha256.lower().strip():
            logger.error(
                f"Archive SHA-256 MISMATCH against pinned expected hash!\n"
                f"  Computed: {computed_archive_hash.lower()}\n"
                f"  Expected: {args.expected_sha256.lower().strip()}"
            )
            sys.exit(1)
        logger.info(f"Archive SHA-256 MATCHES pinned expected hash: {computed_archive_hash}")

    if ledger_path.exists():
        if not verify_archive_ledger(args.archive, ledger_path):
            logger.error("Ledger verification FAILED! Halting restoration.")
            sys.exit(1)
    elif not args.expected_sha256:
        logger.warning(f"No ledger file found at {ledger_path} and no --expected-sha256 provided; proceeding with caution.")

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

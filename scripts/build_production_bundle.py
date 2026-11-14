#!/usr/bin/env python3
"""
SatQuery AI — Production Artifact Bundle Packager
=================================================
Packages the 13 frozen production assets, the freeze manifest, and
classical baseline configurations into a versioned, immutable .tar.gz bundle
for attachment to GitHub Release candidate v1.0.0-rc1.
"""

import hashlib
import json
import logging
import os
import pathlib
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [satquery.pack] %(message)s"
)
logger = logging.getLogger("satquery.pack")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "artifacts" / "deployment" / "m33_production_freeze_manifest.json"


def compute_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_bundle(output_dir: pathlib.Path, version_tag: str = "v1.0.0-rc1"):
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"satquery-production-artifacts-{version_tag}.tar.gz"
    ledger_name = f"{archive_name}.sha256"

    archive_path = output_dir / archive_name
    ledger_path = output_dir / ledger_name

    logger.info(f"Loading freeze manifest from {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Collect required production assets
    bundle_files = [MANIFEST_PATH]
    for key, meta in manifest["artifacts"].items():
        file_path = REPO_ROOT / meta["path"]
        if not file_path.exists():
            raise FileNotFoundError(f"Required production asset {key} missing at {file_path}")
        # Only add files under artifacts/ or data/ (pyproject and requirements are already in Git)
        if meta["path"].startswith("artifacts/") or meta["path"].startswith("data/"):
            bundle_files.append(file_path)

    # 2. Add classical CVA benchmark artifacts required by tests
    additional_test_files = [
        REPO_ROOT / "artifacts" / "specialists" / "change_detection" / "threshold_config.json",
        REPO_ROOT / "artifacts" / "specialists" / "change_detection" / "evaluation_summary.json"
    ]
    for af in additional_test_files:
        if af.exists() and af not in bundle_files:
            bundle_files.append(af)

    logger.info(f"Total files to package: {len(bundle_files)}")

    # 3. Create deterministic tar.gz archive
    with tarfile.open(archive_path, "w:gz") as tar:
        for f in bundle_files:
            rel = f.relative_to(REPO_ROOT)
            arcname = str(rel).replace("\\", "/")
            logger.info(f"  Adding {arcname} ({f.stat().st_size:,} bytes)")
            # Set deterministic metadata
            tarinfo = tar.gettarinfo(str(f), arcname=arcname)
            tarinfo.uid = 1000
            tarinfo.gid = 1000
            tarinfo.uname = "satquery"
            tarinfo.gname = "satquery"
            tarinfo.mtime = 1731556800  # Frozen timestamp (2026-11-14 04:00:00 UTC)
            if tarinfo.isreg():
                with open(f, "rb") as reg_f:
                    tar.addfile(tarinfo, reg_f)
            else:
                tar.addfile(tarinfo)

    archive_size = archive_path.stat().st_size
    archive_sha256 = compute_sha256(archive_path)

    ledger_path.write_text(f"{archive_sha256}  {archive_name}\n", encoding="utf-8")

    logger.info("=" * 60)
    logger.info(f"Bundle successfully created: {archive_path}")
    logger.info(f"Bundle size: {archive_size:,} bytes ({archive_size / (1024*1024):.2f} MiB)")
    logger.info(f"Bundle SHA-256: {archive_sha256}")
    logger.info(f"Ledger file written: {ledger_path}")
    logger.info("=" * 60)

    return archive_path, ledger_path, archive_sha256, archive_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPO_ROOT / "dist_bundle")
    parser.add_argument("--tag", type=str, default="v1.0.0-rc1")
    args = parser.parse_args()
    build_bundle(args.out_dir, args.tag)

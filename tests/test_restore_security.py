"""
SatQuery AI — Release Bundle Restoration Security & Defense-in-Depth Tests
==========================================================================
Durable regression tests verifying that scripts/restore_production_bundle.py
strictly enforces cryptographic integrity, archive immutability, and
rejection of all malicious archive payloads prior to filesystem extraction.
"""

import gzip
import io
import pathlib
import subprocess
import sys
import tarfile
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESTORE_SCRIPT = REPO_ROOT / "scripts" / "restore_production_bundle.py"
BUNDLE_ARCHIVE = REPO_ROOT / "dist_bundle" / "satquery-production-artifacts-v1.0.0-rc1.tar.gz"
PINNED_SHA256 = "97fa1b6d79245a4da92bfd8991d82a3f214adf56a26a5e0355bb2593196bd5c0"


def craft_malicious_archive(archive_path: pathlib.Path, member_type: str, member_name: str, linkname: str = ""):
    """Helper to craft synthetic tar.gz archives with specific attack vectors."""
    with open(archive_path, "wb") as f_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f_out, mtime=1794628800) as gz:
            with tarfile.open(fileobj=gz, mode="w:") as tar:
                # Add one safe member first
                ti_safe = tarfile.TarInfo(name="artifacts/safe_probe.txt")
                ti_safe.size = 5
                ti_safe.mtime = 1794628800
                ti_safe.mode = 0o644
                tar.addfile(ti_safe, io.BytesIO(b"probe"))

                # Add the attack member
                ti = tarfile.TarInfo(name=member_name)
                ti.mtime = 1794628800
                if member_type == "symlink":
                    ti.type = tarfile.SYMTYPE
                    ti.linkname = linkname
                    tar.addfile(ti)
                elif member_type == "hardlink":
                    ti.type = tarfile.LNKTYPE
                    ti.linkname = linkname
                    tar.addfile(ti)
                elif member_type == "fifo":
                    ti.type = tarfile.FIFOTYPE
                    tar.addfile(ti)
                elif member_type == "duplicate":
                    ti.name = "artifacts/safe_probe.txt"
                    ti.size = 6
                    ti.mode = 0o644
                    tar.addfile(ti, io.BytesIO(b"clobbr"))
                else:
                    payload = b"unauthorized payload"
                    ti.size = len(payload)
                    ti.mode = 0o644
                    tar.addfile(ti, io.BytesIO(payload))


def test_valid_bundle_cryptographic_verification(tmp_path):
    """Test that the production bundle verifies cleanly against the pinned SHA-256."""
    assert BUNDLE_ARCHIVE.exists(), f"Production bundle missing at {BUNDLE_ARCHIVE}"
    dest = tmp_path / "restore_dest"
    dest.mkdir()

    cmd = [
        sys.executable,
        str(RESTORE_SCRIPT),
        "--archive", str(BUNDLE_ARCHIVE),
        "--expected-sha256", PINNED_SHA256,
        "--dest", str(dest)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Restoration failed: {res.stderr}"
    assert "Archive SHA-256 MATCHES pinned expected hash" in res.stderr
    assert "13/13 assets passed" in res.stderr


def test_single_byte_tamper_rejection(tmp_path):
    """Test that altering even one byte in the archive causes immediate rejection."""
    assert BUNDLE_ARCHIVE.exists()
    dest = tmp_path / "tamper_dest"
    dest.mkdir()

    # Create a single-byte modified copy
    data = bytearray(BUNDLE_ARCHIVE.read_bytes())
    # Flip one byte near the end
    data[-100] ^= 0xFF
    tampered_archive = tmp_path / "tampered_bundle.tar.gz"
    tampered_archive.write_bytes(data)

    cmd = [
        sys.executable,
        str(RESTORE_SCRIPT),
        "--archive", str(tampered_archive),
        "--expected-sha256", PINNED_SHA256,
        "--dest", str(dest)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0, "Tampered archive was unexpectedly accepted!"
    assert "MISMATCH against pinned expected hash" in res.stderr


def test_traversal_dotdot_rejection(tmp_path):
    """SEC-01: Rejection of path traversal with '..' sequences."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "traversal.tar.gz"
    craft_malicious_archive(archive, "reg", "artifacts/../../escape.txt")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Directory traversal '..' detected" in res.stderr


def test_absolute_path_rejection(tmp_path):
    """SEC-02: Rejection of Unix absolute root paths."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "absolute.tar.gz"
    craft_malicious_archive(archive, "reg", "/etc/shadow")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Absolute path detected" in res.stderr


def test_windows_drive_letter_rejection(tmp_path):
    """SEC-03: Rejection of Windows drive letters."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "windrive.tar.gz"
    craft_malicious_archive(archive, "reg", "C:/Windows/System32/calc.exe")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Windows drive letter detected" in res.stderr


def test_unc_network_path_rejection(tmp_path):
    """SEC-04: Rejection of UNC network share paths."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "unc.tar.gz"
    craft_malicious_archive(archive, "reg", "//attacker.com/share/exploit.dll")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "UNC network path detected" in res.stderr


def test_symlink_directory_escape_rejection(tmp_path):
    """SEC-05: Rejection of symbolic links to prevent directory escape."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "symlink.tar.gz"
    craft_malicious_archive(archive, "symlink", "artifacts/symlink_target", linkname="/etc/passwd")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Member is a symlink or hardlink" in res.stderr


def test_hardlink_clobber_rejection(tmp_path):
    """SEC-06: Rejection of hard links to prevent arbitrary file clobbering."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "hardlink.tar.gz"
    craft_malicious_archive(archive, "hardlink", "artifacts/hardlink_target", linkname="artifacts/safe_probe.txt")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Member is a symlink or hardlink" in res.stderr


def test_special_device_fifo_rejection(tmp_path):
    """SEC-07: Rejection of special device nodes or FIFOs."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "fifo.tar.gz"
    craft_malicious_archive(archive, "fifo", "artifacts/fifo_pipe")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Member is a special device node or FIFO" in res.stderr


def test_duplicate_member_collision_rejection(tmp_path):
    """SEC-08: Rejection of duplicate member entries (collision attacks)."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "duplicate.tar.gz"
    craft_malicious_archive(archive, "duplicate", "")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Duplicate member detected in archive" in res.stderr


def test_out_of_whitelist_workflow_injection_rejection(tmp_path):
    """SEC-09: Rejection of out-of-whitelist paths targeting CI workflows."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "workflow_attack.tar.gz"
    craft_malicious_archive(archive, "reg", ".github/workflows/malicious.yml")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Member outside allowed whitelist" in res.stderr


def test_out_of_whitelist_root_script_rejection(tmp_path):
    """SEC-10: Rejection of out-of-whitelist paths targeting repo root scripts."""
    dest = tmp_path / "extract_dest"
    dest.mkdir()
    archive = tmp_path / "root_script.tar.gz"
    craft_malicious_archive(archive, "reg", "backdoor.py")

    cmd = [sys.executable, str(RESTORE_SCRIPT), "--archive", str(archive), "--dest", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Member outside allowed whitelist" in res.stderr

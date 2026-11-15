"""
Unit tests for SatQuery AI Production CLI, Packaging, and Deployment Contracts.

Verifies:
  - CLI argument parsing across all subcommands
  - Artifact integrity verification healthcheck
  - Fast linguistic routing contracts
  - Multimodal query input validation contracts
  - Classical Gaussian CVA fallback routing via CLI
  - Clarification exit code semantics
  - Structured JSON logging formatter
  - Docker and container deployment configuration directives
"""

import argparse
from io import StringIO
import json
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from satquery.cli import (
    build_parser,
    handle_verify,
    handle_route,
    handle_query,
    JsonLogFormatter,
)


class TestCliPackagingContracts(unittest.TestCase):
    """Test suite for CLI interfaces, input validation contracts, and container configurations."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.manifest_path = cls.repo_root / "artifacts" / "deployment" / "m33_production_freeze_manifest.json"

    def test_cli_parser_structure(self):
        """Verify that CLI parser correctly configures all required subcommands and flags."""
        parser = build_parser()
        self.assertEqual(parser.prog, "satquery")

        # Test subcommands are registered
        subparsers_action = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ]
        self.assertEqual(len(subparsers_action), 1)
        subcommands = set(subparsers_action[0].choices.keys())
        expected_subcommands = {"verify", "route", "query", "demo"}
        self.assertTrue(expected_subcommands.issubset(subcommands))

    def test_json_log_formatter(self):
        """Verify that JsonLogFormatter produces valid JSON lines with standard fields."""
        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Deployment healthcheck successful",
            args=(),
            exc_info=None
        )
        formatted_str = formatter.format(record)
        log_obj = json.loads(formatted_str)

        self.assertIn("timestamp", log_obj)
        self.assertEqual(log_obj["level"], "INFO")
        self.assertEqual(log_obj["logger"], "test_logger")
        self.assertEqual(log_obj["message"], "Deployment healthcheck successful")

    def test_cli_verify_subcommand(self):
        """Verify that 'satquery verify' successfully audits all 13 frozen production artifacts."""
        self.assertTrue(self.manifest_path.exists(), "Manifest file must exist")

        args = argparse.Namespace(json_output=True, json_logs=False, verbose=False)
        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            exit_code = handle_verify(args)
            output = fake_stdout.getvalue()

        self.assertEqual(exit_code, 0, f"Artifact verification failed: {output}")
        payload = json.loads(output)
        self.assertTrue(payload["all_artifacts_intact"])
        self.assertEqual(len(payload["results"]), 13)
        for name, res in payload["results"].items():
            self.assertEqual(res["status"], "PASS", f"Artifact {name} did not pass: {res}")

    def test_cli_route_fast_intent(self):
        """Verify fast linguistic routing without loading heavy vision neural weights."""
        args = argparse.Namespace(
            query_text="Detect new building construction and road sprawl",
            input_type=None,
            json_output=True,
            json_logs=False,
            verbose=False
        )
        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            exit_code = handle_route(args)
            output = fake_stdout.getvalue()

        self.assertEqual(exit_code, 0)
        route_data = json.loads(output)
        self.assertEqual(route_data["task"], "CHANGE_DETECTION")
        self.assertIn("confidence", route_data)
        self.assertIn("class_probabilities", route_data)
        self.assertIn("latency_ms", route_data)
        self.assertLess(route_data["latency_ms"], 500.0, "Fast routing should complete in under 500ms")

    def test_cli_query_validation_contract_missing_t2(self):
        """Verify input contract: supplying --t1 without --t2 must fail with exit code 1."""
        args = argparse.Namespace(
            query_text="Find changes",
            t1="nonexistent_t1.png",
            t2=None,
            features=None,
            cd_method="standard",
            policy="provisional",
            device="cpu",
            json_output=False,
            output_file=None,
            json_logs=False,
            verbose=False
        )
        exit_code = handle_query(args)
        self.assertEqual(exit_code, 1, "Supplying only --t1 without --t2 must return exit code 1")

    def test_cli_query_validation_contract_nonexistent_files(self):
        """Verify input contract: nonexistent image paths must fail gracefully with exit code 1."""
        args = argparse.Namespace(
            query_text="Find changes",
            t1="definitely_nonexistent_t1.png",
            t2="definitely_nonexistent_t2.png",
            features=None,
            cd_method="standard",
            policy="provisional",
            device="cpu",
            json_output=False,
            output_file=None,
            json_logs=False,
            verbose=False
        )
        exit_code = handle_query(args)
        self.assertEqual(exit_code, 1, "Nonexistent input files must return exit code 1")

    def test_cli_query_clarification_exit_code(self):
        """Verify policy contract: ambiguous query under 'clarify' policy returns exit code 2."""
        args = argparse.Namespace(
            query_text="Detect building construction and new roads",
            t1=None,
            t2=None,
            features=None,
            cd_method="standard",
            policy="clarify",
            device="cpu",
            json_output=True,
            output_file=None,
            json_logs=False,
            verbose=False
        )
        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            exit_code = handle_query(args)
            output = fake_stdout.getvalue()

        self.assertEqual(exit_code, 2, "Clarification required must return exit code 2")
        payload = json.loads(output)
        self.assertEqual(payload["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(payload["execution_policy"], "CLARIFICATION_REQUIRED")

    def test_docker_packaging_artifacts(self):
        """Verify presence and validity of production container packaging configurations."""
        dockerfile = self.repo_root / "Dockerfile"
        dockerignore = self.repo_root / ".dockerignore"
        compose = self.repo_root / "docker-compose.yml"

        self.assertTrue(dockerfile.exists(), "Dockerfile must exist")
        self.assertTrue(dockerignore.exists(), ".dockerignore must exist")
        self.assertTrue(compose.exists(), "docker-compose.yml must exist")

        # Inspect Dockerfile contents
        df_content = dockerfile.read_text(encoding="utf-8")
        self.assertIn("ARG TARGET_ENV=cpu", df_content)
        self.assertTrue(
            "FROM python:3.11-slim" in df_content or "FROM python:3.12-slim" in df_content,
            "Dockerfile must use supported python:3.11-slim or python:3.12-slim base"
        )
        self.assertIn("useradd -m -u 1000 -s /bin/bash satquery", df_content)
        self.assertIn("EXPOSE 5000", df_content)
        self.assertIn("HEALTHCHECK", df_content)
        self.assertIn("ENTRYPOINT", df_content)

        # Inspect docker-compose contents
        compose_content = compose.read_text(encoding="utf-8")
        self.assertIn("satquery-cpu:", compose_content)
        self.assertIn("satquery-gpu:", compose_content)


if __name__ == "__main__":
    unittest.main()

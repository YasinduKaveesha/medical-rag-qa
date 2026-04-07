"""Tests for Module 9 — Gradio multimodal tab and ingestion script."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pytest
import yaml


# ---------------------------------------------------------------------------
# test_gradio_has_two_tabs
# ---------------------------------------------------------------------------


def test_gradio_has_two_tabs():
    """build_demo() returns a gr.Blocks with two tabs."""
    from app.gradio_demo import build_demo

    demo = build_demo()
    assert isinstance(demo, gr.Blocks)


# ---------------------------------------------------------------------------
# test_ask_multimodal_function_exists
# ---------------------------------------------------------------------------


def test_ask_multimodal_function_exists():
    """ask_multimodal_question is importable and callable."""
    from app.gradio_demo import ask_multimodal_question
    import inspect

    assert callable(ask_multimodal_question)
    sig = inspect.signature(ask_multimodal_question)
    params = list(sig.parameters.keys())
    assert "question" in params
    assert "use_vision" in params


# ---------------------------------------------------------------------------
# test_ingest_script_importable
# ---------------------------------------------------------------------------


def test_ingest_script_importable():
    """ingest_multimodal.py can be imported and exposes main() and parse_args()."""
    from scripts.ingest_multimodal import main, parse_args

    assert callable(main)
    assert callable(parse_args)

    # parse_args should return namespace with expected attributes
    ns = parse_args([])
    assert hasattr(ns, "pdf_dir")
    assert hasattr(ns, "output_dir")
    assert hasattr(ns, "device")
    assert ns.device == "cpu"


# ---------------------------------------------------------------------------
# test_docker_compose_valid
# ---------------------------------------------------------------------------


def test_docker_compose_valid():
    """docker-compose.yml is valid YAML and contains required services."""
    compose_path = Path(__file__).parent.parent / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml not found"

    with compose_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    services = data["services"]
    assert "qdrant" in services, "Missing 'qdrant' service"
    assert "fastapi" in services, "Missing 'fastapi' service"


# ---------------------------------------------------------------------------
# test_ci_yaml_valid
# ---------------------------------------------------------------------------


def test_ci_yaml_valid():
    """ci.yml is valid YAML, has a test step, and excludes slow/integration tests."""
    ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), ".github/workflows/ci.yml not found"

    with ci_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "jobs" in data

    # Find the pytest command across all job steps
    pytest_cmd = None
    for job in data["jobs"].values():
        for step in job.get("steps", []):
            run_cmd = step.get("run", "")
            if "pytest" in run_cmd:
                pytest_cmd = run_cmd
                break

    assert pytest_cmd is not None, "No pytest step found in ci.yml"
    assert "not slow" in pytest_cmd, "CI should skip slow tests"
    assert "not integration" in pytest_cmd, "CI should skip integration tests"

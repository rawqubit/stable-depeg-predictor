"""Tests for stable-depeg-predictor."""
import sys
import subprocess
import pytest


def test_predictor_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", "predictor.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", "app.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

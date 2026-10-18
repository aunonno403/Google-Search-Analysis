"""Smoke test: the dashboard script loads and renders without raising."""

from streamlit.testing.v1 import AppTest


def test_app_starts_without_errors():
    at = AppTest.from_file("../main.py").run(timeout=30)
    assert not at.exception
    # A SyntaxError doesn't show up in at.exception, just as an empty page.
    assert any("Analyse Trends" in b.label for b in at.button)

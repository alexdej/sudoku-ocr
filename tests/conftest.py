import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_pillow: test requires Pillow for image loading"
        " (pip install 'sudoku-ocr[formats]')",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_pillow"):
        pytest.importorskip(
            "PIL",
            reason="Pillow not installed — pip install 'sudoku-ocr[formats]'",
        )

from pathlib import Path

import pytest


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_pya.db")


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_pya.db")

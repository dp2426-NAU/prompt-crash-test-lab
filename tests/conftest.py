"""Shared test configuration and fixtures."""

import sys
from pathlib import Path

# Ensure project root is in path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

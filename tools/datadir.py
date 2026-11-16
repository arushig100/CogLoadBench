"""Locate the saved data directory.

DATA_ROOT is read once at import, from $COGLOAD_DATA or data/ in the code directory;
the ELS and evaluation modules build their input paths from it.
"""
import os
from pathlib import Path

# Code directory (the parent of tools/).
CODE_ROOT = Path(__file__).resolve().parents[1]
# Message added to errors raised when the saved data is missing.
MISSING_DATA = 'Requires the saved data (cogload-data.tar.gz; see README.md); unpack it into data/ at the repository root or set COGLOAD_DATA to its data/ directory.'


def data_root():
    """Return $COGLOAD_DATA, or data/ in the code directory."""
    return Path(os.environ.get('COGLOAD_DATA', CODE_ROOT / 'data')).expanduser().resolve()


DATA_ROOT = data_root()

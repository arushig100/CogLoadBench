"""Saved ELS inputs and references in the data directory, and shared output-directory helpers."""
import argparse
import json
from pathlib import Path

from tools.datadir import DATA_ROOT, MISSING_DATA

# The paper's saved ELS predictions for Tables 3, 4, D1 and D6, one subdirectory per analysis, read-only.
REFERENCES = DATA_ROOT / 'els/reference'
# Identity, condition and load label of every prompt row that the saved predictions index.
REFERENCE_ROWS = REFERENCES / 'rows.csv.gz'
# The paper's ELS values, compared against recomputed results by els.verify.
REFERENCE_VALUES = REFERENCES / 'reference_values.json'
# A fixed gzip timestamp keeps compressed outputs byte-identical across runs.
GZIP = {'method': 'gzip', 'mtime': 0}


def parser_with_output(description):
    """Argument parser with --output-dir and --features."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--features', type=Path, required=True,
                        help='Feature table written by extraction.build_features')
    return parser


def require_inputs(parser, *paths):
    """Stop with a usage error if an input file is missing."""
    missing = [Path(path) for path in paths if not Path(path).is_file()]
    if missing:
        # Point to the data download when the missing file belongs in the data directory.
        saved = any(path.resolve().is_relative_to(DATA_ROOT) for path in missing)
        parser.error(f"Missing {', '.join(map(str, missing))}." + (f' {MISSING_DATA}' if saved else ''))


def new_output(path):
    """Resolve and create an output directory outside the saved references."""
    output = Path(path).resolve()
    if output.is_relative_to(REFERENCES):
        raise ValueError('Reference outputs are read-only; choose another output directory')
    output.mkdir(parents=True, exist_ok=True)
    return output


def start_output(path, manifest):
    """Create or resume an output directory whose protocol.json equals manifest."""
    output = new_output(path)
    saved = output / 'protocol.json'
    # A nonempty directory is resumed only if it was started with the same protocol.
    if any(output.iterdir()) and not saved.exists():
        raise ValueError('Nonempty output directory has no protocol.json')
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError('Existing outputs use another protocol; use a new output directory')
    saved.write_text(json.dumps(manifest, indent=2) + '\n')
    return output

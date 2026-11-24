"""Create and verify the file inventory of the data directory."""
# An inventory is MANIFEST.tsv at a directory's root, listing every file's relative
# path, size and SHA-256. run.py checks the data directory against its inventory
# before reproduce.
import argparse
import csv
import hashlib
from pathlib import Path
import re
import sys
import tempfile

from tools.datadir import MISSING_DATA

# Version-control, cache and environment folders that are never inventoried.
IGNORED = {'__pycache__', '.git', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.venv', 'venv'}


def sha256(path):
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def files(root):
    """Sorted release files under root, excluding the manifest; rejects symlinks and private files."""
    result = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in IGNORED for part in relative.parts):
            continue
        if path.name == '.DS_Store':
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink is not a release file: {relative}')
        if path.is_file() and relative.as_posix() != 'MANIFEST.tsv':
            # Hidden folders hold local configuration, which never ships in a release.
            if (any(part.startswith('.') for part in relative.parts[:-1])
                    or path.name.startswith('.env') or path.suffix in {'.pem', '.key'}):
                raise ValueError(f'Private file in release: {relative}')
            result.append(path)
    return result


def write(root):
    """Write root/MANIFEST.tsv, replacing any existing one only after the new one is complete."""
    if not root.is_dir():
        raise ValueError(f'Inventory root is not a directory: {root}')
    rows = [[path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)]
            for path in files(root)]
    if not rows:
        raise ValueError(f'Cannot write an empty inventory: {root}')
    # Write to a temporary file in the same directory and rename it into place.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=root,
                                         prefix='.manifest-', delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['path', 'bytes', 'sha256'])
            writer.writerows(rows)
        temporary.chmod(0o644)  # Temporary files are created owner-only.
        temporary.replace(root / 'MANIFEST.tsv')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify(root):
    """Check root's files against MANIFEST.tsv and return the number of files."""
    inventory = root / 'MANIFEST.tsv'
    if not inventory.is_file():
        raise ValueError(f'Missing {inventory}. {MISSING_DATA}')
    with inventory.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != ['path', 'bytes', 'sha256']:
            raise ValueError('Invalid manifest header')
        entries = list(reader)
    # Every row needs a path, a decimal size and a 64-digit lowercase hash.
    # Row numbers count the header as line 1.
    for number, row in enumerate(entries, 2):
        if (set(row) != {'path', 'bytes', 'sha256'}
                or any(not isinstance(value, str) or not value for value in row.values())
                or not row['bytes'].isdecimal()
                or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            raise ValueError(f'Malformed manifest row {number}')
    # The listed paths must equal the files on disk, with no duplicates.
    names = [row['path'] for row in entries]
    actual = {path.relative_to(root).as_posix(): path for path in files(root)}
    if not names or len(set(names)) != len(names):
        raise ValueError('Manifest lists no files or a file twice')
    if set(names) != set(actual):
        missing, extra = sorted(set(names) - set(actual)), sorted(set(actual) - set(names))
        raise ValueError(f'Manifest inventory does not match the files: missing {missing[:5]}, '
                         f'not listed {extra[:5]}')
    for row in entries:
        path = actual[row['path']]
        if int(row['bytes']) != path.stat().st_size or row['sha256'] != sha256(path):
            raise ValueError(f'Changed or damaged file: {row["path"]}')
    return len(entries)


def main():
    """Verify (or, with --write, write and verify) the inventory of a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory whose MANIFEST.tsv to check')
    parser.add_argument('--write', action='store_true', help='Rewrite MANIFEST.tsv before checking it')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.write:
            write(root)
        print(f'PASS: {verify(root)} files match the inventory.')
    except (FileNotFoundError, ValueError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

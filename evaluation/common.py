"""Helpers shared by the evaluation checks."""
from __future__ import annotations

import sys
from typing import Callable


def unique_object(pairs):
    """JSON object hook that rejects duplicate keys instead of keeping the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def run(main: Callable[[], int]) -> None:
    """Exit with main()'s status; a missing or invalid input prints FAIL and exits with status 1."""
    try:
        status = main()
    except (FileNotFoundError, ValueError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        status = 1
    raise SystemExit(status)

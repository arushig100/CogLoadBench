"""Check saved ELS predictions against their prompt rows and recompute their summaries."""
# run.py reproduce runs this with no arguments. $COGLOAD_DATA/els/reference holds the
# paper's saved predictions for Table 3, Table D1, Table 4 and Table D6; the summaries
# recomputed from them must reproduce every value in reference_values.json at its
# listed rounding.
# run.py fit runs it with --fit-dir and --features on a fresh refit of every analysis. The
# cross-axis and calibration protocol.json files must match the feature table, the predictions pass
# the same checks, and each summary recomputed from its saved scores must equal its
# summary.csv.
#
# Checks of a set of saved predictions:
#   1. cross-axis and calibration predictions cover the labeled prompt rows,
#      their test rows share no base problem with their training rows, and their AUCs
#      follow from the saved scores;
#   2. the summaries of every analysis are recomputed from the saved scores.
import argparse
import json
from functools import cache
from pathlib import Path

import pandas as pd

from els import behavior, families, generalization
from els.calibrate import protocol as calibration_protocol
from els.features import aggregated, load_features
from els.fit import protocol as cross_axis_protocol
from els.paths import REFERENCE_ROWS, REFERENCE_VALUES, REFERENCES, require_inputs
from els.validation import (SOURCE_COLUMNS, calibration_summary, cross_axis_summary, rounded_as_listed,
                            require_summary, validate_calibration, validate_cross_axis)
from evaluation.common import run, unique_object

# Analyses with saved references, and the analyses a refit writes, in output order.
REFERENCE_ANALYSES = ['cross_axis', 'calibration', 'generalization', 'families']
FIT_ANALYSES = ['cross_axis', 'calibration', 'generalization', 'behavior', 'families']
# Index columns of each analysis's summary.csv.
SUMMARY_INDEX = {'cross_axis': 0, 'calibration': 0, 'generalization': [0, 1], 'behavior': 0, 'families': [0, 1]}
# Names used in messages.
LABELS = {'cross_axis': 'Table 3 (ELS, length, entropy, margin)', 'calibration': 'Table D1 (leave-one-axis-out calibration)',
          'generalization': 'Table 4 (task and level generalization)', 'behavior': 'Table D2 (behavior-calibrated ELS)',
          'families': 'Tables D6-D7 (feature-family ablations and signatures)'}
# The saved family references hold the ablation scores of Table D6.
REFERENCE_LABELS = {**LABELS, 'families': 'Table D6 (feature-family ablations)'}


@cache
def reference_values():
    """The reference values of each analysis, after checking that every entry is well formed."""
    values = json.loads(REFERENCE_VALUES.read_text(), object_pairs_hook=unique_object)
    if set(values) != set(REFERENCE_ANALYSES):
        raise ValueError(f'{REFERENCE_VALUES} must list exactly {REFERENCE_ANALYSES}')
    for name, entries in values.items():
        for entry in entries:
            columns, digits = entry.get('columns'), entry.get('digits')
            if (not {'table', 'row'} <= entry.keys() or not isinstance(columns, list) or not columns
                    or not isinstance(entry.get('values'), list)
                    or len(entry['values']) != len(columns)
                    or not (isinstance(digits, int) or (isinstance(digits, list) and len(digits) == len(columns)))):
                raise ValueError(f'{REFERENCE_VALUES}: malformed {name} entry {entry}')
    return values


def check_reference(name, tables):
    """Require each reference value of an analysis to equal the recomputed value, rounded to its listed digits."""
    # Each entry names a table, a row and columns, the listed digits, and the values.
    entries = reference_values().get(name)
    if not entries:
        raise ValueError(f'{REFERENCE_VALUES} lists no values for {name}')
    for entry in entries:
        row = tuple(entry['row']) if isinstance(entry['row'], list) else entry['row']
        try:
            computed = tables[entry['table']].loc[row, entry['columns']]
        except KeyError as error:
            raise ValueError(f'{REFERENCE_LABELS[name]}: no recomputed value for {entry["table"]} {row} {entry["columns"]}') from error
        shown = rounded_as_listed(computed, entry['digits'])
        if shown != tuple(entry['values']):
            raise ValueError(f'{REFERENCE_LABELS[name]} {entry["table"]} {row}: {shown} differs from the paper value {tuple(entry["values"])}')


def recompute(root, name, source, data, *, features=True):
    """Recompute one analysis from its saved predictions; return its summary and named tables."""
    output = root / name
    if name == 'cross_axis':
        summary = cross_axis_summary(validate_cross_axis(output, source))
        return summary, {'summary': summary}
    if name == 'calibration':
        summary = calibration_summary(validate_calibration(output, source))
        # summary.csv column headers (the K values) are read back as strings.
        summary.columns = summary.columns.astype(str)
        return summary, {'summary': summary}
    if name == 'generalization':
        summary = generalization.evaluate(output, data)[2]
        return summary, {'summary': summary}
    if name == 'behavior':
        summary = behavior.evaluate(output, data)[3]
        return summary, {'summary': summary}
    if name == 'families':
        labeled = families.labeled(data)
        if not features:
            # The saved references hold the Table D6 ablation scores, which need no features.
            _, ablations = families.evaluate_ablations(output, labeled)
            return ablations, {'ablations': ablations, 'drop_effects': families.drop_effects(ablations)}
        _, signatures, _, ablations = families.evaluate(output, labeled)
        return (families.summary_frame(signatures, ablations),
                {'ablations': ablations, 'drop_effects': families.drop_effects(ablations)})
    raise ValueError(f'Unknown analysis: {name}')


def main():
    """Check the saved references, or with --fit-dir a refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, help='Check a directory written by run.py fit instead')
    parser.add_argument('--features', type=Path, help='Feature table of the refit (required with --fit-dir)')
    args = parser.parse_args()
    root = args.fit_dir.resolve() if args.fit_dir else REFERENCES
    if args.fit_dir:
        if args.features is None:
            parser.error('--fit-dir requires --features')
        require_inputs(parser, args.features, *[root / name / 'protocol.json' for name in ['cross_axis', 'calibration']])
        protocols = {'cross_axis': cross_axis_protocol(args.features), 'calibration': calibration_protocol(args.features)}
        for name, expected in protocols.items():
            path = root / name / 'protocol.json'
            if json.loads(path.read_text()) != expected:
                raise ValueError(f'{path}: protocol or feature-table hash differs')
        # source: identity columns only, for checking saved rows and rebuilding splits;
        # data: the full table with per-layer aggregates, for recomputing the analyses.
        source = pd.read_parquet(args.features, columns=SOURCE_COLUMNS)
        data = aggregated(load_features(args.features))
    else:
        require_inputs(parser, REFERENCE_ROWS, REFERENCE_VALUES)
        # One row per prompt, in the row order that the saved predictions index.
        data = pd.read_csv(REFERENCE_ROWS)
        if list(data.columns) != [*SOURCE_COLUMNS, 'level_index']:
            raise ValueError(f'{REFERENCE_ROWS}: unexpected columns')
        source = data[SOURCE_COLUMNS]
    for name in FIT_ANALYSES if args.fit_dir else REFERENCE_ANALYSES:
        summary, tables = recompute(root, name, source, data, features=bool(args.fit_dir))
        if args.fit_dir:
            saved = pd.read_csv(root / name / 'summary.csv', index_col=SUMMARY_INDEX[name])
            # The generalization, behavior and families summaries have a fixed row order
            # (the families summary also has empty cells), so they are compared in order.
            require_summary(summary, saved, f'{LABELS[name]} summary', ordered=name in ('generalization', 'behavior', 'families'))
            print(f'PASS: {LABELS[name]}: the refit summary follows from its saved predictions.', flush=True)
        else:
            check_reference(name, tables)
            print(f'PASS: {REFERENCE_LABELS[name]}: every value matches the paper.', flush=True)
    return 0


if __name__ == '__main__':
    run(main)

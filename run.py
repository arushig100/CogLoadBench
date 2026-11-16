#!/usr/bin/env python3
"""Run a benchmark task, feature extraction, steering, direction building, an ELS refit, or the reproduction."""
# Single entry point for the release. reproduce and fit are handled here; benchmark,
# extract, steer and directions forward their remaining arguments to the matching
# script. Every step runs in a subprocess; reproduce and fit use single-threaded math
# libraries so that refits are deterministic.
import argparse
import os
from pathlib import Path
import subprocess
import sys

# Keep __pycache__ folders out of the code directory, here and in every subprocess (-B).
sys.dont_write_bytecode = True
from tools.datadir import CODE_ROOT as ROOT, data_root

# CogLoadBench tasks accepted by the benchmark command.
TASKS = ('math', 'context', 'analogy')
# pandas 1.5 calls np.find_common_type, which NumPy 1.25+ deprecates.
QUIET = 'ignore:np.find_common_type:DeprecationWarning'


def execute(arguments, *, environment=None, cwd=ROOT, single_thread=True):
    """Run the current Python on arguments with a fixed environment, raising on failure."""
    warnings = ','.join(filter(None, [QUIET, os.environ.get('PYTHONWARNINGS')]))
    # Single-threaded math libraries keep reproduce and fit deterministic.
    threads = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'} if single_thread else {}
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONWARNINGS': warnings, **threads, **(environment or {})}
    subprocess.run([sys.executable, '-B', *map(str, arguments)], cwd=cwd, env=env, check=True)


def parse_args(argv=None):
    """Parse the command line and reject output directories that could overwrite code or data."""
    # extract and directions pass every argument, including --help, to their script.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {'extract', 'directions'}:
        return argparse.Namespace(command=argv[0], arguments=argv[1:])
    # benchmark and steer need a task or dataset before their script's own --help.
    if argv[:1] in (['benchmark'], ['steer']) and argv[1:2] in (['-h'], ['--help']):
        name, choices = ('task', TASKS) if argv[0] == 'benchmark' else ('dataset', ('cuad', 'qasper'))
        print(f"usage: run.py {argv[0]} {{{','.join(choices)}}} [options]\n"
              f"Run 'python run.py {argv[0]} <{name}> --help' for the {name}'s options.")
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    descriptions = {'reproduce': "Recompute the paper's results from its saved data",
                    'fit': 'Fit every ELS analysis on a feature table'}
    for name in ['reproduce', 'fit']:
        sub = commands.add_parser(name, help=descriptions[name])
        if name == 'reproduce':
            sub.add_argument('--data-dir', type=Path, default=data_root(),
                             help='Saved data directory (default: $COGLOAD_DATA, else data/ in the code directory)')
            sub.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reproduction',
                             help='Directory for the recomputed tables (default: outputs/reproduction)')
        if name == 'fit':
            sub.add_argument('--features', type=Path, required=True,
                             help='Feature table written by extraction.build_features')
            sub.add_argument('--output-dir', type=Path, required=True,
                             help='Directory for the refitted predictions and summaries')
            sub.add_argument('--workers', type=int, default=1,
                             help='Parallel processes for the calibration fits (default: 1)')
    runners = {'benchmark': 'Run a CogLoadBench task (math, context, or analogy)',
               'extract': 'Extract activation features from prompts',
               'steer': 'Run CUAD or QASPER inference, optionally with steering',
               'directions': 'Build steering directions from extracted features'}
    for name in runners:
        sub = commands.add_parser(name, add_help=False, help=runners[name])
        if name == 'benchmark':
            sub.add_argument('task', choices=TASKS)
        if name == 'steer':
            sub.add_argument('dataset', choices=['cuad', 'qasper'])
        sub.add_argument('arguments', nargs=argparse.REMAINDER, metavar='...')
    args = parser.parse_args(argv)
    if args.command == 'fit' and args.workers < 1:
        parser.error('--workers must be positive')
    # Outputs go under outputs/ or to a directory outside both the code and the data.
    if args.command in {'reproduce', 'fit'}:
        output = args.output_dir.resolve()
        data = (args.data_dir if args.command == 'reproduce' else data_root()).resolve()
        if output.is_relative_to(data) or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'outputs')):
            parser.error('--output-dir must be under outputs/ or outside the code and data directories')
        if any(path.exists() and not path.is_dir() for path in (output, *output.parents)):
            parser.error('--output-dir and its existing parents must be directories')
    return args


def main(argv=None):
    """Run the requested command."""
    args = parse_args(argv)
    if args.command == 'fit':
        # Fit every ELS analysis, then check that each summary follows from the saved predictions.
        output, features = args.output_dir.resolve(), args.features.resolve()
        execute(['-m', 'els.fit', '--features', features, '--output-dir', output / 'cross_axis'])
        execute(['-m', 'els.calibrate', '--features', features, '--output-dir', output / 'calibration',
                 '--workers', args.workers])
        for name in ['generalization', 'behavior', 'families']:
            execute(['-m', f'els.{name}', '--features', features, '--output-dir', output / name])
        execute(['-m', 'els.verify', '--fit-dir', output, '--features', features])
    elif args.command == 'reproduce':
        data = args.data_dir.resolve()
        environment = {'COGLOAD_DATA': str(data)}
        # Check every data file against the data inventory before using it.
        from tools.inventory import verify
        verify(data)
        # Rebuild the benchmark tables, then run each verifier; any failure stops the run.
        execute(['-m', 'evaluation.benchmark.compute', '--output-dir', args.output_dir.resolve() / 'benchmark'], environment=environment)
        for module in ['evaluation.benchmark.verify', 'els.verify', 'evaluation.steering.verify']:
            execute(['-m', module], environment=environment)
        print('PASS: benchmark, ELS, and steering results match the paper.')
    else:
        if args.command == 'benchmark':
            script = ROOT / f'benchmark/{args.task}/run_{args.task}.py'
        elif args.command == 'extract':
            script = ROOT / 'extraction/extract_features.py'
        elif args.command == 'steer':
            script = ROOT / f'steering/run_{args.dataset}.py'
        else:
            script = ROOT / 'steering/build_directions.py'
        # Forwarded scripts run from the caller's directory, so relative paths keep their meaning.
        # A leading '--' only separates run.py's arguments from the script's and is dropped.
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        # The code directory goes first on PYTHONPATH so scripts can import the release packages.
        path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))
        environment = {'PYTHONPATH': path}
        # A fixed hash seed keeps steering runs reproducible.
        if args.command == 'steer':
            environment['PYTHONHASHSEED'] = '0'
        execute([script, *arguments], environment=environment, cwd=Path.cwd(), single_thread=False)


if __name__ == '__main__':
    try:
        main()
    # Input errors print one line and exit with status 2; a failed subprocess passes on its status.
    except (ValueError, FileNotFoundError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)

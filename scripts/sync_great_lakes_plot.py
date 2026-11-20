#!/usr/bin/env python3
"""Run on your Mac to mirror Great Lakes PNGs into the local image viewer.

By default, new progress images are discovered on every poll.
One dedicated SSH connection handles authentication for the whole session.
The viewer/Cloudflare tunnel and running Slurm job do not need to restart.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time


DEFAULT_RUN = 'four_laser_inphase_all38_symmetric_values_rms_4ghz_standard_lk_mirrored_1k'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT/'rl/rl_api/runs/great_lakes_live/four_laser_symmetric'
SYMMETRIC_TEST_PLOTS = (
    ('four_sym_A_pairscore_laser1_1k', 'four_A_pairscore_progress.png'),
    ('four_sym_B_directed_meanref_1k', 'four_B_meanref_progress.png'),
    ('four_sym_C_pairscore_meanref_1k', 'four_C_both_progress.png'),
)
FIXED_CONTEXT_TEST_PLOTS = tuple(
    (f'four_fixed_fresh_{span}ghz', f'four_fixed_{span}ghz_progress.png')
    for span in (1, 2, 3, 4))
FIXED_CONTEXT_ASYMMETRIC_TEST_PLOTS = tuple(
    (f'four_fixed_fresh_{span}ghz_asym', f'four_fixed_{span}ghz_progress_asym.png')
    for span in (1, 2, 3, 4))


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='greatlakes.arc-ts.umich.edu',
                        help='SSH host or user@host; your normal SSH configuration applies')
    parser.add_argument('--repository', default='vcsel_lib',
                        help='Remote repository path, relative to your remote home or absolute')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all-runs', action='store_true',
                           help='Discover all run-level progress PNGs every poll (default)')
    selection.add_argument('--run', help='Watch only this training RUN_NAME')
    selection.add_argument('--symmetric-tests', action='store_true',
                           help='Mirror all three named four-laser comparison PNGs in one SSH session')
    selection.add_argument('--fixed-context-tests', action='store_true',
                           help='Mirror the four fresh fixed-context 1/2/3/4 GHz training PNGs')
    parser.add_argument('--include-asymmetric', action='store_true',
                        help='With --fixed-context-tests, also mirror the four _asym plots')
    parser.add_argument('--include-precision', action='store_true',
                        help='With --fixed-context-tests, also mirror the symmetric 4 GHz precision plot')
    parser.add_argument('--job-id', help='Optional exact Slurm job instead of the run-level live image')
    parser.add_argument('--filename', default='current_training_progress.png',
                        help='Progress PNG filename for a single --run')
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--interval', type=float, default=10., help='Seconds between checks (default: 10)')
    parser.add_argument('--once', action='store_true', help='Copy once, then close the dedicated connection')
    parser.add_argument('--dry-run', action='store_true', help='Show paths without connecting or copying')
    args = parser.parse_args(argv)
    args.auto_discover = args.all_runs or not (args.run is not None or args.symmetric_tests or args.fixed_context_tests
        or args.job_id or args.filename != 'current_training_progress.png')
    if args.all_runs and (args.job_id or args.filename != 'current_training_progress.png'):
        parser.error('--all-runs discovers its own images; omit --job-id/--filename')
    args.run = DEFAULT_RUN if args.run is None else args.run
    if (args.include_asymmetric or args.include_precision) and not args.fixed_context_tests:
        parser.error('--include-asymmetric/--include-precision require --fixed-context-tests')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.host):
        parser.error('--host must be an SSH hostname or user@hostname')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run):
        parser.error('--run must be a RUN_NAME, not a path')
    if args.job_id is not None and not re.fullmatch(r'[0-9]+(?:_[0-9]+)?', args.job_id):
        parser.error('--job-id must be a Slurm job number')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.png', args.filename):
        parser.error('--filename must be a PNG filename without a directory')
    if (args.symmetric_tests or args.fixed_context_tests) and (args.job_id or args.filename != 'current_training_progress.png'):
        parser.error('Test suites select their own run-level images; omit --job-id/--filename')
    if not 1. <= args.interval <= 3600.:
        parser.error('--interval must lie between 1 and 3600 seconds')
    return args


def remote_plot(args):
    path = remote_root(args)/args.run
    if args.job_id:
        path /= args.job_id
    return path/args.filename


def remote_plots(args):
    if args.auto_discover:
        return []  # Resolve the current list on the server on every poll.
    if args.symmetric_tests or args.fixed_context_tests:
        root = remote_root(args)
        plots = FIXED_CONTEXT_TEST_PLOTS if args.fixed_context_tests else SYMMETRIC_TEST_PLOTS
        if args.include_asymmetric:
            plots += FIXED_CONTEXT_ASYMMETRIC_TEST_PLOTS
        if args.include_precision:
            plots += (('four_fixed_4ghz_precision', 'four_fixed_4ghz_progress_precision.png'),)
        return [root/run/filename for run, filename in plots]
    return [remote_plot(args)]


def remote_root(args):
    return PurePosixPath(args.repository)/'rl/rl_api/runs/great_lakes'


DISCOVERY_SCRIPT = r'''import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1]).expanduser()
if not root.is_absolute():
    root = pathlib.Path.home()/root
paths = []
if root.is_dir():
    for run in root.iterdir():
        if run.is_dir() and not run.is_symlink() and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', run.name):
            for plot in run.iterdir():
                if (plot.is_file() and not plot.is_symlink() and 'progress' in plot.name.lower()
                        and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.[Pp][Nn][Gg]', plot.name)):
                    paths.append(str(plot.relative_to(root)))
print('VCSEL_PROGRESS_JSON='+json.dumps(sorted(paths)))
'''


def check_connection(host, socket):
    subprocess.run(['ssh', '-S', socket, '-O', 'check', host],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)


def discover_plots(args, socket):
    """Scan only live run-level PNGs, never checkpoints or job histories."""
    check_connection(args.host, socket)
    command = shlex.join(['python3', '-c', DISCOVERY_SCRIPT, str(remote_root(args))])
    result = subprocess.run(['ssh', '-S', socket, '-o', 'ControlMaster=no', '-o', 'BatchMode=yes',
                             '-o', 'ConnectTimeout=10', args.host, command],
                            capture_output=True, text=True, check=True, timeout=30)
    # Login startup files may print banners; parse our explicitly marked line.
    payload = next((line.partition('=')[2] for line in reversed(result.stdout.splitlines())
                    if line.startswith('VCSEL_PROGRESS_JSON=')), None)
    try:
        paths = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError('Could not read the remote progress-image list') from error
    if not isinstance(paths, list):
        raise RuntimeError('Invalid remote progress-image list')
    sources = []
    for name in paths:
        path = PurePosixPath(name) if isinstance(name, str) else None
        if (path is None or path.is_absolute() or len(path.parts) != 2
                or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', path.parts[0])
                or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.[Pp][Nn][Gg]', path.name)
                or 'progress' not in path.name.lower()):
            raise RuntimeError('Invalid remote progress-image path')
        sources.append(remote_root(args)/path)
    return sorted(set(sources))


def discovered_filename(source):
    """Stable, distinct flat names for the viewer, including generic plot names."""
    name = f'{source.parent.name}__{source.name}'
    if len(name.encode()) > 240:
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
        name = name[:210]+'__'+digest+'.png'
    return name


def copy_discovered(args, sources, cache, socket):
    """One incremental rsync for the entire discovered list per poll."""
    check_connection(args.host, socket)
    relative = [str(source.relative_to(remote_root(args))) for source in sources]
    transport = shlex.join(['ssh', '-S', socket, '-o', 'ControlMaster=no', '-o', 'BatchMode=yes'])
    remote = args.host+':'+shlex.quote(str(remote_root(args))+'/')
    result = subprocess.run(['rsync', '-rt', '--relative', '--from0', '--files-from=-', '--timeout=30',
                             '-e', transport, '--', remote, str(cache)+'/'],
                            input='\0'.join(relative)+'\0', capture_output=True, text=True, timeout=45)
    if result.returncode == 0 or result.returncode == 24:
        return
    if result.returncode == 23 and 'No such file' in result.stderr:
        return  # A run can disappear between discovery and transfer.
    raise RuntimeError(result.stderr.strip() or f'rsync failed with exit code {result.returncode}')


def mirror_discovered(args, destination, socket):
    previous = {}
    known = set()
    with tempfile.TemporaryDirectory(prefix='.plot-sync-', dir=destination.parent) as folder:
        cache = Path(folder)
        while True:
            sources = discover_plots(args, socket)
            for source in sources:
                if source not in known:
                    print(f'Discovered {source.parent.name}/{source.name}', flush=True)
                    known.add(source)
            if sources:
                copy_discovered(args, sources, cache, socket)
            available = 0
            for source in sources:
                downloaded = cache/str(source.relative_to(remote_root(args)))
                if not downloaded.is_file():
                    continue
                available += 1
                stat = downloaded.stat()
                current = (stat.st_mtime_ns, stat.st_size)
                image = destination/discovered_filename(source)
                if current != previous.get(source) or not image.exists():
                    temporary = image.with_suffix('.png.tmp')
                    shutil.copy2(downloaded, temporary)
                    temporary.replace(image)
                    previous[source] = current
                    print(time.strftime('%H:%M:%S')+' updated '+image.name, flush=True)
            if args.once:
                return 0 if sources and available == len(sources) else 1
            time.sleep(args.interval)


@contextmanager
def ssh_connection(host):
    # OpenSSH on macOS has a short Unix-socket path limit; /tmp also keeps the
    # socket out of the repository. TemporaryDirectory creates a private folder.
    with tempfile.TemporaryDirectory(prefix='vcsel-plot-', dir='/tmp') as folder:
        socket = str(Path(folder)/'ssh')
        command = ['ssh', '-M', '-S', socket, '-o', 'ControlPersist=no',
                   '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
                   '-o', 'ServerAliveCountMax=3', '-fN', host]
        try:
            print('Connecting to Great Lakes; authenticate normally when prompted.', flush=True)
            subprocess.run(command, check=True)
            yield socket
        finally:
            # This closes only our private master, never the VS Code connection.
            try:
                subprocess.run(['ssh', '-S', socket, '-O', 'exit', host],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass


def copy_plot(host, source, destination, socket):
    # Require reuse of the authenticated connection. A failed master must not
    # trigger repeated interactive MFA requests in the polling loop.
    check_connection(host, socket)
    transport = shlex.join(['ssh', '-S', socket, '-o', 'ControlMaster=no', '-o', 'BatchMode=yes'])
    remote = host+':'+shlex.quote(str(source))
    # rsync's default temporary-file + rename behavior gives the viewer only
    # complete images. Preserve mtime so unchanged plots do not trigger refresh.
    result = subprocess.run(['rsync', '-t', '--timeout=30', '-e', transport,
                             '--', remote, str(destination)+'/'],
                            capture_output=True, text=True, timeout=45)
    if result.returncode == 0:
        return True
    if result.returncode == 23 and 'No such file' in result.stderr:
        return False  # Normal while queued or before the first plot is saved.
    raise RuntimeError(result.stderr.strip() or f'rsync failed with exit code {result.returncode}')


def mirror(args):
    sources = remote_plots(args)
    destination = args.destination.expanduser().resolve()
    if args.auto_discover:
        print(f'Watching {args.host}:{remote_root(args)}/*/*progress*.png', flush=True)
        print('New tests are discovered automatically; local names include the run name.', flush=True)
    for source in sources:
        print(f'Remote plot: {args.host}:{source}', flush=True)
    print(f'Local viewer folder: {destination}', flush=True)
    if args.dry_run:
        return 0
    for command in ('ssh', 'rsync'):
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} must be installed on the Mac')
    destination.mkdir(parents=True, exist_ok=True)
    previous = {}
    waiting = set()
    with ssh_connection(args.host) as socket:
        print('Syncing every update. Leave this terminal running; Ctrl+C stops syncing.', flush=True)
        if args.auto_discover:
            return mirror_discovered(args, destination, socket)
        while True:
            all_available = True
            for source in sources:
                available = copy_plot(args.host, source, destination, socket)
                all_available = all_available and available
                if available:
                    image = destination/source.name
                    stat = image.stat()
                    current = (stat.st_mtime_ns, stat.st_size)
                    if current != previous.get(source):
                        print(time.strftime('%H:%M:%S')+' updated '+image.name, flush=True)
                        previous[source] = current
                    waiting.discard(source)
                elif source not in waiting:
                    print(f'Waiting for the cluster to write {source.name}...', flush=True)
                    waiting.add(source)
            if args.once:
                return 0 if all_available else 1
            time.sleep(args.interval)


def main(argv=None):
    args = options(argv)
    try:
        return mirror(args)
    except KeyboardInterrupt:
        print('\nPlot sync stopped. Training and the image viewer are still running.', flush=True)
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f'Plot sync stopped: {error}\nRun this command again to reconnect.', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

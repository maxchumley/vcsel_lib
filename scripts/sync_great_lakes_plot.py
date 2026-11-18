#!/usr/bin/env python3
"""Run on your Mac to mirror Great Lakes PNGs into the local image viewer.

One dedicated SSH connection handles authentication for the whole session.
The viewer/Cloudflare tunnel and running Slurm job do not need to restart.
"""
import argparse
from contextlib import contextmanager
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


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='greatlakes.arc-ts.umich.edu',
                        help='SSH host or user@host; your normal SSH configuration applies')
    parser.add_argument('--repository', default='vcsel_lib',
                        help='Remote repository path, relative to your remote home or absolute')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--run', default=DEFAULT_RUN, help='Training RUN_NAME')
    selection.add_argument('--symmetric-tests', action='store_true',
                           help='Mirror all three named four-laser comparison PNGs in one SSH session')
    selection.add_argument('--fixed-context-tests', action='store_true',
                           help='Mirror the four fresh fixed-context 1/2/3/4 GHz training PNGs')
    parser.add_argument('--job-id', help='Optional exact Slurm job instead of the run-level live image')
    parser.add_argument('--filename', default='current_training_progress.png',
                        help='Progress PNG filename for a single --run')
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--interval', type=float, default=10., help='Seconds between checks (default: 10)')
    parser.add_argument('--once', action='store_true', help='Copy once, then close the dedicated connection')
    parser.add_argument('--dry-run', action='store_true', help='Show paths without connecting or copying')
    args = parser.parse_args(argv)
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
    path = PurePosixPath(args.repository)/'rl/rl_api/runs/great_lakes'/args.run
    if args.job_id:
        path /= args.job_id
    return path/args.filename


def remote_plots(args):
    if args.symmetric_tests or args.fixed_context_tests:
        root = PurePosixPath(args.repository)/'rl/rl_api/runs/great_lakes'
        plots = FIXED_CONTEXT_TEST_PLOTS if args.fixed_context_tests else SYMMETRIC_TEST_PLOTS
        return [root/run/filename for run, filename in plots]
    return [remote_plot(args)]


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
    subprocess.run(['ssh', '-S', socket, '-O', 'check', host],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
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

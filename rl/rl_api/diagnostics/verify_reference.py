"""Check the frozen recipe without starting training.

python -m rl.rl_api.diagnostics.verify_reference
python -m rl.rl_api.diagnostics.verify_reference --physics

--physics also compares two real 500 ns trajectories against a direct backend
call. This uses a throwaway initialized policy, not the live checkpoint.
"""
from dataclasses import asdict, replace
import numpy as np
import torch

from rl.rl_api import CouplingModel, Experiment
from rl.rl_api._compat import core, reference_settings
from rl.rl_api.models import build_policy
from rl.rl_api.simulation import numpy_seed


def verify_physics(config, *, parallel=False):
    """Compare all simulator arrays exactly, then both reward calculations."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        policy = build_policy(config)
    model = CouplingModel(policy, config)
    designs = model.propose(target_phases_rad=[0., -1., 2.],
        detunings_ghz=[-.5, 0., .5], count=2, seed=7)
    evaluated = designs.evaluate(seed=123)
    detuning = np.repeat(designs.context.detunings_ghz[None], len(designs), axis=0)
    with numpy_seed(123):
        direct = core.simulate_with_vcsel(designs.coupling_strengths_per_ns,
            designs.coupling_phases_rad, designs.config,
            detuning_distribution_ghz=detuning, progress=False, smooth_frequencies=True)
    for actual, expected in zip(evaluated.simulation, direct):
        np.testing.assert_array_equal(actual, expected)
    targets = np.repeat(designs.context.target_phases_rad[None], len(designs), axis=0)
    cosine = core.calculate_phase_reward(*direct[:2], targets, designs.config)[0]
    rms = core.calculate_phase_rms_metrics(*direct[:2], targets, designs.config)["phase_rms_reward"]
    np.testing.assert_array_equal(evaluated.cosine_reward, cosine)
    np.testing.assert_array_equal(evaluated.rms_reward, rms)
    print("PASS: two real 500 ns trajectories and both rewards match exactly.")
    if parallel:
        serial = core.simulate_candidates_parallel(targets,
            designs.coupling_strengths_per_ns[:, None], designs.coupling_phases_rad[:, None],
            replace(designs.config, n_jobs=1), detuning_distributions_ghz=detuning)
        workers = core.simulate_candidates_parallel(targets,
            designs.coupling_strengths_per_ns[:, None], designs.coupling_phases_rad[:, None],
            replace(designs.config, n_jobs=2), detuning_distributions_ghz=detuning)
        for key in serial:
            np.testing.assert_array_equal(serial[key], workers[key])
        print("PASS: two spawned simulation workers match serial scoring exactly.")


def verify(*, physics=False, parallel=False):
    """Read-only configuration/source check; raises on a numerical mismatch."""
    experiment = Experiment("reference_check")
    experiment.describe()
    actual, frozen = asdict(experiment.backend_config), reference_settings()
    output_paths = {f"{kind}_checkpoint_file" for kind in ("best", "current", "final")}
    # JSON stores tuples as arrays; normalize both sides before comparison.
    import json
    actual = json.loads(json.dumps(actual))
    differences = [key for key in frozen if key not in output_paths and actual[key] != frozen[key]]
    if differences:
        raise AssertionError(f"Configuration differences: {differences}")
    print(f"PASS: all {len(frozen)-len(output_paths)} stored non-path configuration fields match.")
    status = experiment.verify_reference()
    for name, matches in status.items():
        print(f"{'MATCH' if matches else 'CHANGED'}: {name}")
    required = ("vcsel_lib.py", "rl/conditional_coupling_bernoulli_sparse.py",
                "rl/conditional_coupling_designer_variable_m.py")
    if not all(status[name] for name in required):
        raise AssertionError("Numerical reference files changed; rerun parity checks")
    if physics or parallel:
        verify_physics(experiment.backend_config, parallel=parallel)
    return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Also compare two spawned simulation workers")
    args = parser.parse_args()
    verify(physics=args.physics, parallel=args.parallel)

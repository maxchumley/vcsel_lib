"""Display-only coupling-edge selection regression tests."""

from rl.conditional_coupling_run_variable_m import sample_edges_across_strengths


def test_strength_spread_includes_weak_middle_and_strong_edges():
    edges = [(index, index + 1, float(index)) for index in range(10)]
    selected = sample_edges_across_strengths(edges, 3, lambda edge: edge[2])
    assert [edge[2] for edge in selected] == [0.0, 4.0, 9.0]


def test_strength_spread_is_deterministic_with_equal_strengths():
    edges = [(index, index + 1, 1.0) for index in reversed(range(5))]
    selected = sample_edges_across_strengths(edges, 3, lambda edge: edge[2])
    assert [(source, receiver) for source, receiver, _ in selected] == [
        (0, 1), (2, 3), (4, 5),
    ]

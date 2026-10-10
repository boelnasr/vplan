from scripts.spikes.cylinder_contact import cylinder_seeds


def test_cylinder_benchmark_selects_the_documented_cohorts() -> None:
    first = cylinder_seeds(range(60))
    comparison = cylinder_seeds(range(60, 144))
    assert len(first) == 21  # seed 50 is subsequently blocked by the Verifier
    assert (first[0], first[-1]) == (5, 58)
    assert len(comparison) == 45
    assert (comparison[0], comparison[-1]) == (60, 143)

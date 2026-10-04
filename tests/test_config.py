from vplan.config import config_hash, config_snapshot, load_config


def test_load_all_configs_have_expected_keys() -> None:
    assert load_config("verifier")["collision_margin_m"] == 0.01
    assert load_config("judge")["rest_dwell_s"] == 0.5
    assert load_config("physics")["lateral_friction"] == 0.8


def test_snapshot_has_all_three() -> None:
    assert set(config_snapshot()) == {"verifier", "judge", "physics"}


def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_verifier_and_judge_use_the_same_jacobian_length() -> None:  # D23.5: sigma_min is only comparable for equal L
    assert load_config("verifier")["jacobian_char_length_m"] == load_config("judge")["jacobian_char_length_m"]

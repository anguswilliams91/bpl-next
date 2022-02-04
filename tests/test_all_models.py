"""Shared tests across all models, e.g. checking probabilities are valid."""
import jax.numpy as jnp
import pytest

from bpl import DixonColesMatchPredictor, ExtendedDixonColesMatchPredictor
from bpl.base import MAX_GOALS

MODELS = [DixonColesMatchPredictor, ExtendedDixonColesMatchPredictor]


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_score_proba(dummy_data, model_cls):

    model = model_cls().fit(dummy_data, num_samples=100, num_warmup=100)

    probs = model.predict_score_proba(
        dummy_data["home_team"],
        dummy_data["away_team"],
        dummy_data["home_goals"],
        dummy_data["away_goals"],
    )

    assert jnp.all((probs >= 0) & (probs <= 1))

    prob_single = model.predict_score_proba("0", "1", 1, 0)[0]
    assert 0 <= prob_single <= 1


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_outcome_proba(dummy_data, model_cls):
    model = model_cls().fit(dummy_data, num_samples=100, num_warmup=100)

    probs = model.predict_outcome_proba(
        dummy_data["home_team"], dummy_data["away_team"]
    )

    total_probability = probs["home_win"] + probs["away_win"] + probs["draw"]

    assert jnp.allclose(total_probability, 1.0, atol=1e-5)

    prob_single = model.predict_outcome_proba("0", "1")
    assert prob_single["home_win"] + prob_single["away_win"] + prob_single[
        "draw"
    ] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_score_n_proba(dummy_data, model_cls):
    model = model_cls().fit(dummy_data, num_samples=100, num_warmup=100)

    n = jnp.arange(MAX_GOALS + 1)
    proba_home = model.predict_score_n_proba(n, "0", "1")
    assert len(proba_home) == len(n)
    assert jnp.all((proba_home >= 0) & (proba_home <= 1))
    assert sum(proba_home) == pytest.approx(1.0, abs=1e-5)

    proba_away = model.predict_score_n_proba(n, "0", "1", home=False)
    assert len(proba_home) == len(n)
    assert jnp.all((proba_away >= 0) & (proba_away <= 1))
    assert sum(proba_away) == pytest.approx(1.0, abs=1e-5)

    assert sum(proba_home * n) > sum(proba_away * n)  # score more at home

    proba_single = model.predict_score_n_proba(1, "0", "1")
    assert len(proba_single) == 1
    assert (proba_single[0] >= 0) and (proba_single[0] <= 1)


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_concede_n_proba(dummy_data, model_cls):
    model = model_cls().fit(dummy_data, num_samples=100, num_warmup=100)

    n = jnp.arange(MAX_GOALS + 1)
    proba_home = model.predict_concede_n_proba(n, "0", "1")
    assert len(proba_home) == len(n)
    assert jnp.all((proba_home >= 0) & (proba_home <= 1))
    assert sum(proba_home) == pytest.approx(1.0, abs=1e-5)

    proba_away = model.predict_concede_n_proba(n, "0", "1", home=False)
    assert len(proba_home) == len(n)
    assert jnp.all((proba_away >= 0) & (proba_away <= 1))
    assert sum(proba_away) == pytest.approx(1.0, abs=1e-5)

    assert sum(proba_home * n) < sum(proba_away * n)  # concede more away

    proba_team_concede = model.predict_concede_n_proba(1, "0", "1")
    assert len(proba_team_concede) == 1
    assert (proba_team_concede[0] >= 0) and (proba_team_concede[0] <= 1)

    proba_opponent_score = model.predict_score_n_proba(1, "1", "0", home=False)
    assert proba_team_concede == pytest.approx(proba_opponent_score, abs=1e-5)

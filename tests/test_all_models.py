"""Shared tests across all models, e.g. checking probabilities are valid."""
import jax.numpy as jnp
import pytest

from bpl.dixon_coles import DixonColesMatchPredictor

MODELS = [DixonColesMatchPredictor]


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


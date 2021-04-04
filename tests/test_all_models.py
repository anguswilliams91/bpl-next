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

    assert jnp.all(0 <= probs <= 1)

    prob_single = model.predict_score_proba("A", "B", 1, 0)[0]
    assert 0 <= prob_single <= 1


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_outcome_proba(dummy_data, model_cls):
    model = model_cls().fit(dummy_data, num_samples=100, num_warmup=100)

    probs = model.predict_outcome_proba(
        dummy_data["home_team"], dummy_data["away_team"]
    )

    assert jnp.all(0 <= probs <= 1)

    prob_single = model.predict_outcome_proba("A", "B")
    assert 0 <= prob_single <= 1

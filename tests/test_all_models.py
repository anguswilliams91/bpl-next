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

    proba_home = jnp.array(
        [
            model.predict_score_n_proba(n, "0", "1")
            for n in range(MAX_GOALS + 1)
        ]
    )
    assert jnp.all((proba_home >= 0) & (proba_home <= 1))
    assert sum(proba_home) == pytest.approx(1.0, abs=1e-5)

    proba_away = jnp.array(
        [
            model.predict_score_n_proba(n, "0", "1", home=False)
            for n in range(MAX_GOALS + 1)
        ]
    )
    assert jnp.all((proba_away >= 0) & (proba_away <= 1))
    assert sum(proba_away) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_concede_n_proba(dummy_data, model_cls):
    model = model_cls().fit(dummy_data, num_samples=100, num_warmup=100)

    proba_home = jnp.array(
        [
            model.predict_concede_n_proba(n, "0", "1")
            for n in range(MAX_GOALS + 1)
        ]
    )
    assert jnp.all((proba_home >= 0) & (proba_home <= 1))
    assert sum(proba_home) == pytest.approx(1.0, abs=1e-5)

    proba_away = jnp.array(
        [
            model.predict_concede_n_proba(n, "0", "1", home=False)
            for n in range(MAX_GOALS + 1)
        ]
    )
    assert jnp.all((proba_away >= 0) & (proba_away <= 1))
    assert sum(proba_away) == pytest.approx(1.0, abs=1e-5)

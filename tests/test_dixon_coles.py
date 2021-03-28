import jax.numpy as jnp
import pytest

from bpl.dixon_coles import DixonColesMatchPredictor


@pytest.fixture
def fitted_model(dummy_data):
    return DixonColesMatchPredictor().fit(dummy_data, num_samples=100, num_warmup=100)


def test_fit(dummy_data):
    model = DixonColesMatchPredictor().fit(dummy_data)

    assert model.attack is not None
    assert model.defence is not None
    assert model.home_advantage is not None
    assert model.teams is not None
    assert model.corr_coef is not None


def test_predict_score_proba(dummy_data, fitted_model):

    probs = fitted_model.predict_score_proba(
        dummy_data["home_team"],
        dummy_data["away_team"],
        dummy_data["home_goals"],
        dummy_data["away_goals"],
    )

    assert jnp.all((probs >= 0.0) <= (probs <= 1.0))

    prob_single = fitted_model.predict_score_proba("A", "B", 1, 0)[0]
    assert (prob_single >= 0) and (prob_single <= 1)

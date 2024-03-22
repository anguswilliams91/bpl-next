import jax.numpy as jnp
import pytest

from bpl.base import MAX_GOALS
from bpl.neutral_dixon_coles import NeutralDixonColesMatchPredictor

TOL = 1e-02


@pytest.fixture
def model(neutral_dummy_data):
    return NeutralDixonColesMatchPredictor().fit(neutral_dummy_data)


def test_fit(model):
    assert model.attack is not None
    assert model.defence is not None
    assert model.home_attack is not None
    assert model.home_defence is not None
    assert model.away_attack is not None
    assert model.away_defence is not None
    assert model.teams is not None
    assert model.corr_coef is not None


def test_predict_score_proba(model, neutral_dummy_data):
    probs = model.predict_score_proba(
        neutral_dummy_data["home_team"],
        neutral_dummy_data["away_team"],
        neutral_dummy_data["home_goals"],
        neutral_dummy_data["away_goals"],
        neutral_dummy_data["neutral_venue"],
    )

    assert jnp.all((probs >= 0) & (probs <= 1))

    prob_single = model.predict_score_proba("0", "1", 1, 0, 0)[0]
    assert 0 <= prob_single <= 1


def test_predict_outcome_proba(model, neutral_dummy_data):
    probs = model.predict_outcome_proba(
        neutral_dummy_data["home_team"],
        neutral_dummy_data["away_team"],
        neutral_dummy_data["neutral_venue"],
    )

    total_probability = probs["home_win"] + probs["away_win"] + probs["draw"]

    assert jnp.allclose(total_probability, 1.0, atol=TOL)

    prob_single = model.predict_outcome_proba("0", "1", 0)
    assert prob_single["home_win"] + prob_single["away_win"] + prob_single[
        "draw"
    ] == pytest.approx(1.0, abs=TOL)


def test_predict_score_n_proba(model):
    n = jnp.arange(MAX_GOALS + 1)
    proba_home = model.predict_score_n_proba(n, "0", "1")
    assert len(proba_home) == len(n)
    assert jnp.all((proba_home >= 0) & (proba_home <= 1))
    assert sum(proba_home) == pytest.approx(1.0, abs=TOL)

    proba_away = model.predict_score_n_proba(n, "0", "1", home=False)
    assert len(proba_away) == len(n)
    assert jnp.all((proba_away >= 0) & (proba_away <= 1))
    assert sum(proba_away) == pytest.approx(1.0, abs=TOL)

    assert sum(proba_home * n) > sum(proba_away * n)  # score more at home

    proba_single = model.predict_score_n_proba(1, "0", "1")
    assert len(proba_single) == 1
    assert (proba_single[0] >= 0) and (proba_single[0] <= 1)


def test_predict_concede_n_proba(model):
    n = jnp.arange(MAX_GOALS + 1)
    proba_home = model.predict_concede_n_proba(n, "0", "1")
    assert len(proba_home) == len(n)
    assert jnp.all((proba_home >= 0) & (proba_home <= 1))
    assert sum(proba_home) == pytest.approx(1.0, abs=TOL)

    proba_away = model.predict_concede_n_proba(n, "0", "1", home=False)
    assert len(proba_away) == len(n)
    assert jnp.all((proba_away >= 0) & (proba_away <= 1))
    assert sum(proba_away) == pytest.approx(1.0, abs=TOL)

    assert sum(proba_home * n) < sum(proba_away * n)  # concede more away

    proba_team_concede = model.predict_concede_n_proba(1, "0", "1")
    assert len(proba_team_concede) == 1
    assert (proba_team_concede[0] >= 0) and (proba_team_concede[0] <= 1)

    proba_opponent_score = model.predict_score_n_proba(1, "1", "0", home=False)
    assert proba_team_concede.tolist() == pytest.approx(
        proba_opponent_score.tolist(), abs=TOL
    )


def test_predict_outcome_neutral_proba(model, neutral_dummy_data):
    probs = model.predict_outcome_proba(
        neutral_dummy_data["home_team"],
        neutral_dummy_data["away_team"],
        neutral_dummy_data["neutral_venue"],
    )

    neutral_home_win = float(
        probs["home_win"][neutral_dummy_data["neutral_venue"] == 1].mean()
    )
    neutral_away_win = float(
        probs["away_win"][neutral_dummy_data["neutral_venue"] == 1].mean()
    )
    normal_home_win = float(
        probs["home_win"][neutral_dummy_data["neutral_venue"] == 0].mean()
    )
    normal_away_win = float(
        probs["away_win"][neutral_dummy_data["neutral_venue"] == 0].mean()
    )

    assert normal_home_win > normal_away_win
    assert normal_home_win > neutral_home_win
    assert neutral_away_win > normal_away_win

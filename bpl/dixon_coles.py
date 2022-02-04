"""Implementation of a simple team level model."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

from bpl._util import dixon_coles_correlation_term
from bpl.base import BaseMatchPredictor

__all__ = ["DixonColesMatchPredictor"]


class DixonColesMatchPredictor(BaseMatchPredictor):
    """A Dixon-Coles like model for predicting match outcomes."""

    def __init__(self):
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.corr_coef = None

    # pylint: disable=too-many-locals
    @staticmethod
    def _model(
        home_team: jnp.array,
        away_team: jnp.array,
        num_teams: int,
        home_goals: Iterable[float],
        away_goals: Iterable[float],
    ):
        std_attack = numpyro.sample("std_attack", dist.HalfNormal(1.0))
        std_defence = numpyro.sample("std_defence", dist.HalfNormal(1.0))
        mean_defence = numpyro.sample("mean_defence", dist.Normal(0.0, 1.0))
        home_advantage = numpyro.sample("home_advantage", dist.Normal(0.1, 0.2))
        corr_coef = numpyro.sample("corr_coef", dist.Normal(0.0, 1.0))

        with numpyro.plate("teams", num_teams):
            with reparam(
                config={
                    "attack": LocScaleReparam(centered=0),
                    "defence": LocScaleReparam(centered=0),
                }
            ):
                attack = numpyro.sample("attack", dist.Normal(0.0, std_attack))
                defence = numpyro.sample(
                    "defence", dist.Normal(mean_defence, std_defence)
                )

        expected_home_goals = jnp.exp(
            attack[home_team] - defence[away_team] + home_advantage
        )
        expected_away_goals = jnp.exp(attack[away_team] - defence[home_team])

        numpyro.sample(
            "home_goals", dist.Poisson(expected_home_goals).to_event(1), obs=home_goals
        )
        numpyro.sample(
            "away_goals", dist.Poisson(expected_away_goals).to_event(1), obs=away_goals
        )

        corr_term = dixon_coles_correlation_term(
            home_goals, away_goals, expected_home_goals, expected_away_goals, corr_coef
        )
        numpyro.factor("correlation_term", corr_term.sum(axis=-1))

    # pylint: disable=arguments-differ,too-many-arguments
    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        random_state: int = 42,
        num_warmup: int = 500,
        num_samples: int = 1000,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DixonColesMatchPredictor:

        home_team = training_data["home_team"]
        away_team = training_data["away_team"]

        self.teams = sorted(list(set(home_team) | set(away_team)))
        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, num_warmup, num_samples, **(mcmc_kwargs or {}))
        rng_key = jax.random.PRNGKey(random_state)
        mcmc.run(
            rng_key,
            home_ind,
            away_ind,
            len(self.teams),
            np.array(training_data["home_goals"]),
            np.array(training_data["away_goals"]),
            **(run_kwargs or {}),
        )

        samples = mcmc.get_samples()
        self.attack = samples["attack"]
        self.defence = samples["defence"]
        self.home_advantage = samples["home_advantage"]
        self.corr_coef = samples["corr_coef"]

        return self

    def _calculate_expected_goals(
        self, home_team: Union[str, Iterable[str]], away_team: Union[str, Iterable[str]]
    ) -> (jnp.array, jnp.array):

        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        attack_home, defence_home = self.attack[:, home_ind], self.defence[:, home_ind]
        attack_away, defence_away = self.attack[:, away_ind], self.defence[:, away_ind]

        home_rate = jnp.exp(attack_home - defence_away + self.home_advantage[:, None])
        away_rate = jnp.exp(attack_away - defence_home)

        return home_rate, away_rate

    def predict_score_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_goals: Union[float, Iterable[float]],
        away_goals: Union[float, Iterable[float]],
    ) -> jnp.array:

        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

        expected_home_goals, expected_away_goals = self._calculate_expected_goals(
            home_team, away_team
        )
        corr_term = dixon_coles_correlation_term(
            home_goals,
            away_goals,
            expected_home_goals,
            expected_away_goals,
            self.corr_coef,
        )

        home_probs = jnp.exp(dist.Poisson(expected_home_goals).log_prob(home_goals))
        away_probs = jnp.exp(dist.Poisson(expected_away_goals).log_prob(away_goals))

        sampled_probs = jnp.exp(corr_term) * home_probs * away_probs
        return sampled_probs.mean(axis=0)

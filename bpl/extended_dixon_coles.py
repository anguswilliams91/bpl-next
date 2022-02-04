"""Implementation of the model in the current version of bpl."""
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

__all__ = ["ExtendedDixonColesMatchPredictor"]


class ExtendedDixonColesMatchPredictor(BaseMatchPredictor):
    """A Dixon-Coles like model for predicting match outcomes."""

    def __init__(self):
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.corr_coef = None
        self.rho = None
        self.attack_coefficients = None
        self.defence_coefficients = None

    # pylint: disable=too-many-locals
    @staticmethod
    def _model(
        home_team: jnp.array,
        away_team: jnp.array,
        num_teams: int,
        home_goals: Iterable[float],
        away_goals: Iterable[float],
        team_covariates: Optional[np.array],
    ):
        std_home_advantage = numpyro.sample(
            "std_home_advantage", dist.HalfNormal(scale=1.0)
        )
        std_attack = numpyro.sample("std_attack", dist.HalfNormal(scale=1.0))
        std_defence = numpyro.sample("std_defence", dist.HalfNormal(scale=1.0))
        mean_defence = numpyro.sample("mean_defence", dist.Normal(loc=0.0, scale=1.0))
        corr_coef = numpyro.sample("corr_coef", dist.Normal(0.0, 1.0))

        mean_home_advantage = numpyro.sample(
            "mean_home_advantage", dist.Normal(0.1, 0.2)
        )

        u = numpyro.sample("u", dist.Beta(concentration1=2.0, concentration0=4.0))
        rho = numpyro.deterministic("rho", 2.0 * u - 1.0)

        if team_covariates is not None:
            standardised_covariates = (
                team_covariates - team_covariates.mean(axis=0)
            ) / team_covariates.std(axis=0)
            num_covariates = standardised_covariates.shape[1]

            with numpyro.plate("covariates", num_covariates):
                attack_coefficients = numpyro.sample(
                    "attack_coefficients", dist.Normal(loc=0.0, scale=1.0)
                )
                defence_coefficients = numpyro.sample(
                    "defence_coefficients", dist.Normal(loc=0.0, scale=1.0)
                )

            attack_prior_mean = jnp.matmul(
                standardised_covariates, attack_coefficients[..., None]
            ).squeeze(-1)
            defence_prior_mean = mean_defence + jnp.matmul(
                standardised_covariates, defence_coefficients[..., None]
            ).squeeze(-1)
        else:
            attack_prior_mean = 0.0
            defence_prior_mean = mean_defence

        with numpyro.plate("teams", num_teams):
            standardised_attack = numpyro.sample(
                "standardised_attack", dist.Normal(loc=0.0, scale=1.0)
            )
            standardised_defence = numpyro.sample(
                "standardised_defence",
                dist.Normal(
                    loc=rho * standardised_attack, scale=jnp.sqrt(1.0 - rho**2.0)
                ),
            )

            with reparam(config={"home_advantage": LocScaleReparam(centered=0)}):
                home_advantage = numpyro.sample(
                    "home_advantage",
                    dist.Normal(mean_home_advantage, std_home_advantage),
                )

        attack = numpyro.deterministic(
            "attack", attack_prior_mean + standardised_attack * std_attack
        )
        defence = numpyro.deterministic(
            "defence", defence_prior_mean + standardised_defence * std_defence
        )

        expected_home_goals = jnp.exp(
            attack[home_team] - defence[away_team] + home_advantage[home_team]
        )
        expected_away_goals = jnp.exp(attack[away_team] - defence[home_team])

        # FIXME: this is because the priors allow crazy simulated data before inference
        expected_home_goals = jnp.clip(expected_home_goals, a_max=15.0)
        expected_away_goals = jnp.clip(expected_away_goals, a_max=15.0)

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
    ) -> ExtendedDixonColesMatchPredictor:

        home_team = training_data["home_team"]
        away_team = training_data["away_team"]
        team_covariates = training_data.get("team_covariates", None)

        self.teams = sorted(list(set(home_team) | set(away_team)))
        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        if team_covariates:
            if set(team_covariates.keys()) == set(self.teams):
                team_covariates = jnp.array([team_covariates[t] for t in self.teams])
            else:
                raise ValueError(
                    "team_covariates must contain all the teams in the data."
                )

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
            team_covariates=team_covariates,
            **(run_kwargs or {}),
        )

        samples = mcmc.get_samples()
        self.attack = samples["attack"]
        self.defence = samples["defence"]
        self.home_advantage = samples["home_advantage"]
        self.corr_coef = samples["corr_coef"]
        self.rho = samples["rho"]
        self.attack_coefficients = samples.get("attack_coefficients", None)
        self.defence_coefficients = samples.get("defence_coefficients", None)

        return self

    def _calculate_expected_goals(
        self, home_team: Union[str, Iterable[str]], away_team: Union[str, Iterable[str]]
    ) -> (jnp.array, jnp.array):

        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        attack_home, defence_home = self.attack[:, home_ind], self.defence[:, home_ind]
        attack_away, defence_away = self.attack[:, away_ind], self.defence[:, away_ind]

        home_rate = jnp.exp(
            attack_home - defence_away + self.home_advantage[:, home_ind]
        )
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

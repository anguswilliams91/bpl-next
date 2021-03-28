"""Implementation of a simple team level model."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS


from bpl.base import BaseMatchPredictor


def _correlation_term(home_goals, away_goals, home_rate, away_rate, corr_coef):
    # correlation term from dixon and coles paper
    corr_term = jnp.zeros_like(home_rate)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    corr_term = jax.ops.index_update(
        corr_term,
        (..., nil_nil),
        jnp.log(1.0 - corr_coef * home_rate[..., nil_nil] * away_rate[..., nil_nil]),
    )

    one_nil = (home_goals == 1) & (away_goals == 0)
    corr_term = jax.ops.index_update(
        corr_term, (..., one_nil), jnp.log(1.0 + corr_coef * away_rate[..., one_nil])
    )

    nil_one = (home_goals == 0) & (away_goals == 1)
    corr_term = jax.ops.index_update(
        corr_term, (..., nil_one), jnp.log(1.0 + corr_coef * home_rate[..., nil_one])
    )

    one_one = (home_goals == 1) & (away_goals == 1)
    corr_term = jax.ops.index_update(
        corr_term, (..., one_one), jnp.log(1.0 - corr_coef)
    )

    return corr_term.sum(axis=-1)


class DixonColesMatchPredictor(BaseMatchPredictor):
    """A Dixon-Coles like model for predicting match outcomes."""

    def __init__(self):
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.corr_coef = None


    def _model(
        self,
        home_team: jnp.array,
        away_team: jnp.array,
        num_teams: int,
        home_goals: Optional[Iterable[float]],
        away_goals: Optional[Iterable[float]],
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

        corr_term = _correlation_term(
            home_goals, away_goals, expected_home_goals, expected_away_goals, corr_coef
        )
        numpyro.factor("correlation_term", corr_term)

    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        random_state: int = 42,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DixonColesMatchPredictor:
        home_team = training_data["home_team"]
        away_team = training_data["away_team"]

        self.teams = sorted(list(set(home_team) | set(away_team)))
        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, **(mcmc_kwargs or {}))
        rng_key = jax.random.PRNGKey(random_state)
        mcmc.run(
            rng_key,
            home_ind,
            away_ind,
            len(self.teams),
            jnp.array(training_data["home_goals"],
            jnp.array(training_data["away_goals"],
            **(run_kwargs or {})
        )

        samples = mcmc.get_samples()
        self.attack = samples["attack"]
        self.defence = samples["defence"]
        self.home_advantage = samples["home_advantage"]
        self.corr_coef = samples["corr_coef"]

        return self

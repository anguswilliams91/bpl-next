"""Implementation of the model in the current version of bpl."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

from bpl_next._util import (
    compute_corr_coef_bounds,
    dixon_coles_correlation_term,
    parse_teams,
)
from bpl_next.base import DTYPES, BaseMatchPredictor

__all__ = ["ExtendedDixonColesMatchPredictor"]


# pylint: disable=too-many-instance-attributes
class ExtendedDixonColesMatchPredictor(BaseMatchPredictor):
    """
    A Dixon-Coles like model for predicting match outcomes, modified to:
    - Estimate correlation between defence and attack abilities
        - strong defenders tend to also be strong attackers
    - Add a separate home advantage for each team (not just a single global parameter)
    - Add option to include team covariates to build informative attack/defence priors
        - should improve initial predictions for new teams (e.g., due to promotion)
          which mostly rely on priors
    - Add option to exponentially downweigh games with time (i.e., recent games ge
      more weight)

    Note: the model can be used to model/predict a single match or a list of matches,
    which means input arrays (e.g., name of home/away teams and number of goals scored)
    have length = number of matches modelled/predicted.
    """

    # pylint: disable=duplicate-code
    def __init__(self):
        super().__init__()
        # attributes get populated when self.fit() is called

        # MCMC samples for each model parameter
        # attack/defence/home_advantage have shape [number of samples, number of teams]
        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.corr_coef = None
        self.rho = None
        # attack/defence_coefficients have shape
        # [number of samples, number of team_covariates]
        self.attack_coefficients = None
        self.defence_coefficients = None
        self.mean_defence = None
        self.std_defence = None
        self.std_attack = None
        self.mean_home_advantage = None
        self.std_home_advantage = None

        # mean and std of covariates (use for standardization)
        self._team_covariates_mean = None
        self._team_covariates_std = None

        # optional time weighting parameter
        self.epsilon = None
        self.time_diff = None

    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
    @staticmethod
    def _model(
        home_team: jnp.array,
        away_team: jnp.array,
        num_teams: int,
        home_goals: Iterable[int],
        away_goals: Iterable[int],
        team_covariates: Optional[np.array],
        time_diff: Optional[Iterable[float]],
        epsilon: Optional[float],
    ):
        """
        NumPyro model definition.

        Args:
            home_team jnp.array: integer indicator of the home team for each match.
            away_team jnp.array: integer indicator of the away team for each match.
            num_teams int: number of teams playing.
            home_goals Iterable[int]: number of goals scored by the home team in each
            match.
            away_goals Iterable[int]: number of goals scored by the away team in each
            match.
            team_covariates Optional[np.array]: optional team covariates
            [num_teams, num_covariates].
            epsilon Optional[float]: optional exponential time decay parameter.
            time_diff Optional[Iterable[float]]: optional number of weeks between
            current game week and match week (must be provided if epsilon is provided).
        """
        # default prior parameters for attack/defence/home_advantage
        mean_attack = 0
        mean_home_advantage = numpyro.sample(
            "mean_home_advantage", dist.Normal(0.1, 0.2)
        )
        std_home_advantage = numpyro.sample(
            "std_home_advantage", dist.HalfNormal(scale=1.0)
        )
        mean_defence = numpyro.sample("mean_defence", dist.Normal(loc=0.0, scale=1.0))
        std_attack = numpyro.sample("std_attack", dist.HalfNormal(scale=1.0))
        std_defence = numpyro.sample("std_defence", dist.HalfNormal(scale=1.0))

        # if have team covariates, build informative attack/defence prior means for each team
        # else use same default prior for all teams
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
            attack_prior_mean = mean_attack
            defence_prior_mean = mean_defence

        # rho parameter accounts for relationship between attack and defence ability
        # (strong attackers tend to also be strong defenders)
        # specify prior on u rather than rho directly (beta constraints u to [0,1] so
        # -1 <= rho <= 1)
        u = numpyro.sample("u", dist.Beta(concentration1=2.0, concentration0=4.0))
        rho = numpyro.deterministic("rho", 2.0 * u - 1.0)

        # estimate attack/defence/home advantage parameters separately for each team
        # - numpyro.plate ensures we get as many parameters as there are teams
        # note we use non centered reparametrisation of all 3 parameters to improve
        # inference
        with numpyro.plate("teams", num_teams):
            # assume for each team rho correlated attack/defence abilities:
            #   (standardised_attack, standardised_defence) ~
            #         Normal([0, 0],[[1, rho], [rho, 1]])
            # below samples standardised_attack and then standardised_defence
            # conditioned on this value
            standardised_attack = numpyro.sample(
                "standardised_attack", dist.Normal(loc=0.0, scale=1.0)
            )
            # note if rho=0, below reduces to N(0, 1)
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
        # transform attack/defence parameters back to centered parametrisation
        # (this is done automatically for home_advantage with LocScaleReparam)
        attack = numpyro.deterministic(
            "attack", attack_prior_mean + standardised_attack * std_attack
        )
        defence = numpyro.deterministic(
            "defence", defence_prior_mean + standardised_defence * std_defence
        )

        # all of above is specified on the log scale so exponentiate to get
        # poisson lambda rate parameters for the home and away teams
        expected_home_goals = jnp.exp(
            attack[home_team] - defence[away_team] + home_advantage[home_team]
        )
        expected_away_goals = jnp.exp(attack[away_team] - defence[home_team])

        # FIXME: this is because the priors allow crazy simulated data before inference
        expected_home_goals = jnp.clip(expected_home_goals, a_max=15.0)
        expected_away_goals = jnp.clip(expected_away_goals, a_max=15.0)

        # likelihood (with optional decaying weights i.e., weigh recent data more
        # heavily)
        if epsilon is not None:
            weights = jnp.exp(-epsilon * time_diff)
            with numpyro.plate("data", len(home_goals)), numpyro.handlers.scale(
                scale=weights
            ):
                numpyro.sample(
                    "home_goals", dist.Poisson(expected_home_goals), obs=home_goals
                )
                numpyro.sample(
                    "away_goals", dist.Poisson(expected_away_goals), obs=away_goals
                )
        else:
            weights = None
            numpyro.sample(
                "home_goals",
                dist.Poisson(expected_home_goals).to_event(1),
                obs=home_goals,
            )
            numpyro.sample(
                "away_goals",
                dist.Poisson(expected_away_goals).to_event(1),
                obs=away_goals,
            )

        # lastly, apply correction for low score matches (tau in Dixon & Coles paper,
        # corr_coeff=rho)
        # ensure the correlation coefficient is within the expected range set out in
        # the Dixon and Coles paper
        corr_coef_raw = numpyro.sample(
            "corr_coef_raw", dist.Beta(concentration1=2.0, concentration0=2.0)
        )

        LB, UB = compute_corr_coef_bounds(expected_home_goals, expected_away_goals)
        corr_coef = numpyro.deterministic("corr_coef", LB + corr_coef_raw * (UB - LB))
        corr_term = dixon_coles_correlation_term(
            home_goals,
            away_goals,
            expected_home_goals,
            expected_away_goals,
            corr_coef,
            weights,
        )
        # numpyro.factor adds log probability to target density
        numpyro.factor("correlation_term", corr_term.sum(axis=-1))

    # pylint: disable=arguments-differ,too-many-arguments,duplicate-code
    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        random_state: int = 42,
        num_warmup: int = 500,
        num_samples: int = 1000,
        epsilon: Optional[float] = None,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ExtendedDixonColesMatchPredictor:
        """
        Fit model to data.
        """
        # prepare data
        self.teams, self._teams_dict, home_ind, away_ind = parse_teams(
            training_data["home_team"], training_data["away_team"], DTYPES["teams"]
        )
        team_covariates = training_data.get("team_covariates", None)

        self.epsilon = epsilon
        self.time_diff = training_data.get("time_diff", None)
        if epsilon is not None:
            if self.time_diff is None:
                raise ValueError(
                    "time_diff must be provided in training_data to include "
                    "exponential time decay in model."
                )

        # if team_covariates are passed, construct informative attack/defence priors
        if team_covariates:
            if set(team_covariates.keys()) == set(self.teams):
                team_covariates = jnp.array([team_covariates[t] for t in self.teams])
                self._team_covariates_mean = team_covariates.mean(axis=0)
                self._team_covariates_std = team_covariates.std(axis=0)
            else:
                raise ValueError(
                    "team_covariates must contain all the teams in the data."
                )

        # initialize model and inference algorithm
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            **(mcmc_kwargs or {}),
        )

        # fit model to data
        rng_key = jax.random.PRNGKey(random_state)
        mcmc.run(
            rng_key,
            # _model are parameters passed here
            home_ind,
            away_ind,
            len(self.teams),
            np.array(training_data["home_goals"]),
            np.array(training_data["away_goals"]),
            team_covariates=team_covariates,
            time_diff=self.time_diff,
            epsilon=self.epsilon,
            **(run_kwargs or {}),
        )

        # save posterior samples
        samples = mcmc.get_samples()
        self.attack = samples["attack"]
        self.defence = samples["defence"]
        self.home_advantage = samples["home_advantage"]
        self.corr_coef = samples["corr_coef"]
        self.rho = samples["rho"]
        self.attack_coefficients = samples.get("attack_coefficients", None)
        self.defence_coefficients = samples.get("defence_coefficients", None)
        self.mean_defence = samples["mean_defence"]
        self.std_defence = samples["std_defence"]
        self.std_attack = samples["std_attack"]
        self.mean_home_advantage = samples["mean_home_advantage"]
        self.std_home_advantage = samples["std_home_advantage"]

        return self

    def _calculate_expected_goals(
        self, home_team: Union[str, Iterable[str]], away_team: Union[str, Iterable[str]]
    ) -> Tuple[jnp.array, jnp.array]:
        """
        Calculate expected goals for home and away team(s) by match.

        Args:
            home_team Union[str, Iterable[str]]: name of home team(s) for each match.
            away_team Union[str, Iterable[str]]: name of away team(s) for each  match.

        Returns:
            Iterable[float], Iterable[float]: expected goals for (home, away) team(s)
            for each match.
        """
        home_ind, away_ind = self._parse_fixture_args(home_team, away_team)

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
        home_goals: Union[int, Iterable[int]],
        away_goals: Union[int, Iterable[int]],
    ) -> jnp.array:
        """
        Return the probability of a particular scoreline.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            home_goals (Union[int, Iterable[int]]): number of goals scored by
                the home team(s).
            away_goals (Union[int, Iterable[int]]): number of goals scored by
                the away team(s).

        Returns:
            float: the probability of the given outcome.
        """
        home_team, away_team = self._parse_fixture_args(home_team, away_team)

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

    def add_new_team(
        self, team_name: str, team_covariates: Optional[np.array] = None
    ) -> None:
        """
        Build defence/attack/home_advantage parameters for team not seen in the training
        data. These are built from default priors unless team covariates are passed.

        Args:
            team_name str: name of new team.
            team_covariates Optional[np.array]: optional team covariates
            [num_teams, num_covariates].
        """
        if team_name in self.teams:
            raise ValueError(f"Team {team_name} already known to model.")

        # can only use team_covariates if coefficients for these were estimated during
        # training if available, build informative priors, else use defaults
        if self.attack_coefficients is not None:
            if team_covariates is None:
                warnings.warn(
                    f"You haven't provided features for {team_name}."
                    " Assuming team_covariates are the average of known teams."
                    " For better forecasts, provide team_covariates."
                )
                team_covariates = jnp.zeros(self.attack_coefficients.shape[1])
            else:
                team_covariates = (
                    0.5
                    * (team_covariates - self._team_covariates_mean)
                    / self._team_covariates_std
                )
            mean_attack = jnp.dot(self.attack_coefficients, team_covariates.ravel())
            mean_defence = self.mean_defence + jnp.dot(
                self.defence_coefficients, team_covariates.ravel()
            )
        else:
            mean_attack = 0.0
            mean_defence = self.mean_defence

        log_a_tilde = np.random.normal(loc=0.0, scale=1.0, size=len(self.std_attack))
        log_b_tilde = np.random.normal(
            loc=self.rho * log_a_tilde, scale=np.sqrt(1 - self.rho**2.0)
        )
        home_advantage = np.random.normal(
            loc=self.mean_home_advantage, scale=self.std_home_advantage
        )

        attack = mean_attack + log_a_tilde * self.std_attack
        defence = mean_defence + log_b_tilde * self.std_defence

        self.teams = np.append(self.teams, team_name)
        self._teams_dict[team_name] = len(self._teams_dict)
        self.attack = jnp.concatenate((self.attack, attack[:, None]), axis=1)
        self.defence = jnp.concatenate((self.defence, defence[:, None]), axis=1)
        self.home_advantage = jnp.concatenate(
            (self.home_advantage, home_advantage[:, None]), axis=1
        )

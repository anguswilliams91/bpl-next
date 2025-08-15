"""Implementation of the neutral model for predicting the World Cup."""

from __future__ import annotations

import warnings
from datetime import datetime
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
    map_choice,
    parse_teams,
    str_to_list,
)
from bpl_next.base import DTYPES, MAX_GOALS

__all__ = ["NeutralDixonColesMatchPredictor"]


# pylint: disable=too-many-instance-attributes
class NeutralDixonColesMatchPredictor:
    """
    A Dixon-Coles like model for predicting match outcomes, modified to:
    - Estimate correlation between defence and attack abilities
        - strong defenders tend to also be strong attackers
    - Add a separate home advantage for each team
      (not just a single global parameter)
    - Add option to include team covariates to build informative
      attack/defence priors
        - should improve initial predictions for new teams (e.g., due to
          promotion) which mostly rely on priors
    - Work for matches in neutral venues (e.g. international tournaments)
    - Add separate home & away, defence & attack, advantages/disadvantages
      for each team
    - Add option to exponentially downweigh games with time
      (i.e., recent games get more weight)
    - Add possibility to weight games according to match importance

    Note that this is a special case of NeutralDixonColesMatchPredictorWC
    model where confederation (or league) strength is not modelled
    (all teams are assumed to be in the same confederation or league).
    """

    # pylint: disable=duplicate-code
    def __init__(self):
        # attributes get populated when self.fit() is called

        # list of all unique team names
        # used to create integer indicator for each team
        self.teams = None
        self._teams_dict = None

        # MCMC samples for each model parameter
        # attack/defence/home_advantage have shape [number of samples, number of teams]
        self.attack = None
        self.defence = None
        self.home_attack = None
        self.away_attack = None
        self.home_defence = None
        self.away_defence = None
        self.time_diff = None
        self.epsilon = None
        self.game_weights = None
        self.corr_coef = None
        self.u = None
        self.rho = None
        # attack/defence_coefficients have shape [number of samples, number of team_covariates]
        self.attack_coefficients = None
        self.defence_coefficients = None
        self.mean_attack = None
        self.mean_defence = None
        self.std_attack = None
        self.std_defence = None
        self.mean_home_attack = None
        self.mean_away_attack = None
        self.mean_home_defence = None
        self.mean_away_defence = None
        self.std_home_attack = None
        self.std_away_attack = None
        self.std_home_defence = None
        self.std_away_defence = None
        self.standardised_attack = None
        self.standardised_defence = None

        # mean and std of covariates (use for standardization)
        self._team_covariates_mean = None
        self._team_covariates_std = None

    # pylint: disable=too-many-arguments,too-many-statements,too-many-locals,duplicate-code
    @staticmethod
    def _model(
        home_team: jnp.array,
        away_team: jnp.array,
        num_teams: int,
        home_goals: Iterable[int],
        away_goals: Iterable[int],
        neutral_venue: Iterable[int],
        time_diff: Optional[Iterable[float]],
        epsilon: Optional[float],
        game_weights: Optional[Iterable[float]],
        team_covariates: Optional[np.array] = None,
    ):
        """
        NumPyro model definition.

        Args:
            home_team jnp.array: integer indicator of the home team for each match.
            away_team jnp.array: integer indicator of the away team for each match.
            num_teams int: number of teams playing.
            home_goals Iterable[int]: goals scored by the home team in each match.
            away_goals Iterable[int]: goals scored by the away team in each match.
            time_diff Optional[Iterable[float]]: optional time difference between
                now and when the game was played (must be provided if epsilon is).
            epsilon Optional[float]: optional exponential time decay parameter.
            game_weights Optional[Iterable[float]]: weights for each game.
            team_covariates Optional[np.array]: optional team covariates
                [num_teams, num_covariates].
        """
        # default prior parameters for attack/defence/home_advantage
        mean_attack = 0.0
        mean_defence = numpyro.sample("mean_defence", dist.Normal(loc=0.0, scale=1.0))
        std_attack = numpyro.sample("std_attack", dist.HalfNormal(scale=0.5))
        std_defence = numpyro.sample("std_defence", dist.HalfNormal(scale=0.5))
        mean_home_attack = numpyro.sample("mean_home_attack", dist.Normal(0.1, 0.2))
        mean_away_attack = numpyro.sample("mean_away_attack", dist.Normal(-0.1, 0.2))
        mean_home_defence = numpyro.sample("mean_home_defence", dist.Normal(0.1, 0.2))
        mean_away_defence = numpyro.sample("mean_away_defence", dist.Normal(-0.1, 0.2))
        std_home_attack = numpyro.sample("std_home_attack", dist.HalfNormal(scale=1.0))
        std_away_attack = numpyro.sample("std_away_attack", dist.HalfNormal(scale=1.0))
        std_home_defence = numpyro.sample(
            "std_home_defence", dist.HalfNormal(scale=1.0)
        )
        std_away_defence = numpyro.sample(
            "std_away_defence", dist.HalfNormal(scale=1.0)
        )

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
        # specify prior on u rather than rho directly (beta constraints u to [0,1]
        # so -1 <= rho <= 1)
        u = numpyro.sample("u", dist.Beta(concentration1=2.0, concentration0=4.0))
        rho = numpyro.deterministic("rho", 2.0 * u - 1.0)

        # estimate attack/defence/home advantage parameters separately for each team
        # - numpyro.plate ensures we get as many parameters as there are teams
        # note we use non centered reparametrisation of all 3 parameters to improve
        # inference
        with numpyro.plate("teams", num_teams):
            # assume for each team rho correlated attack/defence abilities:
            # (standardised_att, standardised_def) ~
            #   Normal([0, 0], [[1, rho], [rho, 1]])
            # below samples standardised_attack and then standardised_defence
            # conditioned on this value
            standardised_attack = numpyro.sample(
                "standardised_attack", dist.Normal(loc=0.0, scale=1.0)
            )
            standardised_defence = numpyro.sample(
                "standardised_defence",
                dist.Normal(
                    loc=rho * standardised_attack, scale=jnp.sqrt(1.0 - rho**2.0)
                ),
            )
            with reparam(config={"home_attack": LocScaleReparam(centered=0)}):
                home_attack = numpyro.sample(
                    "home_attack",
                    dist.Normal(mean_home_attack, std_home_attack),
                )
            with reparam(config={"away_attack": LocScaleReparam(centered=0)}):
                away_attack = numpyro.sample(
                    "away_attack",
                    dist.Normal(mean_away_attack, std_away_attack),
                )
            with reparam(config={"home_defence": LocScaleReparam(centered=0)}):
                home_defence = numpyro.sample(
                    "home_defence",
                    dist.Normal(mean_home_defence, std_home_defence),
                )
            with reparam(config={"away_defence": LocScaleReparam(centered=0)}):
                away_defence = numpyro.sample(
                    "away_defence",
                    dist.Normal(mean_away_defence, std_away_defence),
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
            attack[home_team]
            - defence[away_team]
            + (1 - neutral_venue) * home_attack[home_team]
            - (1 - neutral_venue) * away_defence[away_team]
        )
        expected_away_goals = jnp.exp(
            attack[away_team]
            - defence[home_team]
            + (1 - neutral_venue) * away_attack[away_team]
            - (1 - neutral_venue) * home_defence[home_team]
        )

        # likelihood (with optional decaying weights i.e.,
        # weigh recent data more heavily and according to game_weights)
        weights = jnp.ones(len(home_goals))
        if epsilon is not None:
            weights = weights * jnp.exp(-epsilon * time_diff)
        if weights is not None:
            weights = weights * game_weights

        with numpyro.plate("data", len(home_goals)), numpyro.handlers.scale(
            scale=weights
        ):
            numpyro.sample(
                "home_goals", dist.Poisson(expected_home_goals), obs=home_goals
            )
            numpyro.sample(
                "away_goals", dist.Poisson(expected_away_goals), obs=away_goals
            )

        # impose bounds on the correlation coefficient
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
        numpyro.factor("correlation_term", corr_term.sum(axis=-1))

    # pylint: disable=arguments-differ,too-many-arguments,too-many-statements,duplicate-code
    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        epsilon: Optional[float] = None,
        random_state: int = 42,
        num_warmup: int = 500,
        num_samples: int = 1000,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NeutralDixonColesMatchPredictor:
        """
        Fit model to data.
        """
        # prepare data
        self.teams, self._teams_dict, home_ind, away_ind = parse_teams(
            training_data["home_team"], training_data["away_team"], DTYPES["teams"]
        )
        team_covariates = training_data.get("team_covariates")

        self.epsilon = epsilon
        self.time_diff = training_data.get("time_diff", None)
        if epsilon is not None:
            if self.time_diff is None:
                raise ValueError(
                    """
                    time_diff must be provided in training_data
                    to include exponential time decay in model.
                    """
                )
        self.game_weights = training_data.get("game_weights", None)

        # if team_covariates are passed, construct informative attack/defence priors
        if team_covariates:
            if set(team_covariates.keys()) != set(self.teams):
                raise ValueError(
                    "team_covariates must contain all the teams in the data."
                )
            team_covariates = jnp.array([team_covariates[t] for t in self.teams])
            self._team_covariates_mean = team_covariates.mean(axis=0)
            self._team_covariates_std = team_covariates.std(axis=0)

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
            home_ind,
            away_ind,
            len(self.teams),
            np.array(training_data["home_goals"]),
            np.array(training_data["away_goals"]),
            np.array(training_data["neutral_venue"]),
            self.time_diff,
            self.epsilon,
            self.game_weights,
            team_covariates=team_covariates,
            **(run_kwargs or {}),
        )

        # save posterior samples
        samples = mcmc.get_samples()
        self.attack = samples["attack"]
        self.defence = samples["defence"]
        self.home_attack = samples["home_attack"]
        self.away_attack = samples["away_attack"]
        self.home_defence = samples["home_defence"]
        self.away_defence = samples["away_defence"]
        self.corr_coef = samples["corr_coef"]
        self.u = samples["u"]
        self.rho = samples["rho"]
        self.attack_coefficients = samples.get("attack_coefficients", None)
        self.defence_coefficients = samples.get("defence_coefficients", None)
        # self.mean_attack = samples["mean_attack"]
        self.mean_defence = samples["mean_defence"]
        self.std_attack = samples["std_attack"]
        self.std_defence = samples["std_defence"]
        self.mean_home_attack = samples["mean_home_attack"]
        self.mean_away_attack = samples["mean_away_attack"]
        self.mean_home_defence = samples["mean_home_defence"]
        self.mean_away_defence = samples["mean_away_defence"]
        self.std_home_attack = samples["std_home_attack"]
        self.std_home_defence = samples["std_home_defence"]
        self.std_away_attack = samples["std_away_attack"]
        self.std_away_defence = samples["std_away_defence"]
        self.standardised_attack = samples["standardised_attack"]
        self.standardised_defence = samples["standardised_defence"]

        return self

    def _parse_fixture_args(self, home_team, away_team, neutral_venue):
        home_team, away_team = str_to_list(home_team, away_team)
        neutral_venue = jnp.array(neutral_venue, DTYPES["venue"])
        if isinstance(home_team[0], str):
            home_team = jnp.array(
                [self._teams_dict[t] for t in home_team], DTYPES["teams"]
            )
        if isinstance(away_team[0], str):
            away_team = jnp.array(
                [self._teams_dict[t] for t in away_team], DTYPES["teams"]
            )
        return home_team, away_team, neutral_venue

    def _calculate_expected_goals(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
    ) -> Tuple[jnp.array, jnp.array]:
        """Computes the rate (mean) for the Poisson distribution to model
        the goals scored by home_team and away_team.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.

        Returns:
            Tuple[jnp.array, jnp.array]: Tuple of arrays for home and away rates.
        """
        (
            home_team,
            away_team,
            neutral_venue,
        ) = self._parse_fixture_args(home_team, away_team, neutral_venue)

        attack_home, defence_home = (
            self.attack[:, home_team],
            self.defence[:, home_team],
        )
        attack_away, defence_away = (
            self.attack[:, away_team],
            self.defence[:, away_team],
        )

        home_rate = jnp.exp(
            attack_home
            - defence_away
            + (1 - neutral_venue) * self.home_attack[:, home_team]
            - (1 - neutral_venue) * self.away_defence[:, away_team]
        )
        away_rate = jnp.exp(
            attack_away
            - defence_home
            + (1 - neutral_venue) * self.away_attack[:, away_team]
            - (1 - neutral_venue) * self.home_defence[:, home_team]
        )
        return home_rate, away_rate

    def predict_score_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_goals: Union[int, Iterable[int]],
        away_goals: Union[int, Iterable[int]],
        neutral_venue: Union[int, Iterable[int]],
    ) -> jnp.array:
        """Compute probability of a particular scoreline between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            home_goals (Union[int, Iterable[int]]): number of goals scored by the home team(s).
            away_goals (Union[int, Iterable[int]]): number of goals scored by the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.

        Returns:
            jnp.array: Array of probabilities of each scoreline.
        """
        (
            home_team,
            away_team,
            neutral_venue,
        ) = self._parse_fixture_args(home_team, away_team, neutral_venue)

        expected_home_goals, expected_away_goals = self._calculate_expected_goals(
            home_team, away_team, neutral_venue
        )
        corr_term = dixon_coles_correlation_term(
            home_goals,
            away_goals,
            expected_home_goals,
            expected_away_goals,
            self.corr_coef,
        )

        home_probs = dist.Poisson(expected_home_goals).log_prob(home_goals)
        away_probs = dist.Poisson(expected_away_goals).log_prob(away_goals)

        sampled_probs = jnp.exp(corr_term + home_probs + away_probs)
        return sampled_probs.mean(axis=0)

    def add_new_team(self, team_name: str, team_covariates: Optional[np.array] = None):
        """Method for adding another team to the model.

        Args:
            team_name (str): team name
            team_covariates (Optional[np.array], optional): team covariates to
                initialise prior distribution. Defaults to None.

        Raises:
            ValueError: if `team_name` is already known to the model.
        """
        if team_name in self.teams:
            raise ValueError(f"Team {team_name} already known to model.")

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
        home_attack = np.random.normal(
            loc=self.mean_home_attack, scale=self.std_home_attack
        )
        away_attack = np.random.normal(
            loc=self.mean_away_attack, scale=self.std_away_attack
        )
        home_defence = np.random.normal(
            loc=self.mean_home_defence, scale=self.std_home_defence
        )
        away_defence = np.random.normal(
            loc=self.mean_away_defence, scale=self.std_away_defence
        )
        attack = mean_attack + log_a_tilde * self.std_attack
        defence = mean_defence + log_b_tilde * self.std_defence

        self.teams = np.append(self.teams, team_name)
        self._teams_dict[team_name] = len(self._teams_dict)
        self.attack = jnp.concatenate((self.attack, attack[:, None]), axis=1)
        self.defence = jnp.concatenate((self.defence, defence[:, None]), axis=1)
        self.home_attack = jnp.concatenate(
            (self.home_attack, home_attack[:, None]), axis=1
        )
        self.away_attack = jnp.concatenate(
            (self.away_attack, away_attack[:, None]), axis=1
        )
        self.home_defence = jnp.concatenate(
            (self.home_defence, home_defence[:, None]), axis=1
        )
        self.away_defence = jnp.concatenate(
            (self.away_defence, away_defence[:, None]), axis=1
        )

    def predict_score_grid_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
        max_goals: Optional[int] = MAX_GOALS,
    ) -> Tuple[jnp.array, np.array, np.array]:
        """Calculate scoreline probabilities between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            Tuple[jnp.array, np.array, np.array]: Tuple of the following grids (as arrays):
                probability of scorelines, home goals and away goals scored grids
        """
        (
            home_team,
            away_team,
            neutral_venue,
        ) = self._parse_fixture_args(home_team, away_team, neutral_venue)

        n_goals = np.arange(0, max_goals + 1)
        home_goals, away_goals = np.meshgrid(n_goals, n_goals, indexing="ij")
        home_goals_flat = jnp.tile(
            home_goals.reshape((max_goals + 1) ** 2), len(home_team)
        )
        away_goals_flat = jnp.tile(
            away_goals.reshape((max_goals + 1) ** 2), len(home_team)
        )
        home_team_rep = np.repeat(home_team, (max_goals + 1) ** 2)
        away_team_rep = np.repeat(away_team, (max_goals + 1) ** 2)
        neutral_venue_rep = np.repeat(neutral_venue, (max_goals + 1) ** 2)

        probs = self.predict_score_proba(
            home_team_rep,
            away_team_rep,
            home_goals_flat,
            away_goals_flat,
            neutral_venue_rep,
        ).reshape(len(home_team), max_goals + 1, max_goals + 1)
        return probs, home_goals, away_goals

    def predict_outcome_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
        knockout: bool = False,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> Dict[str, jnp.array]:
        """Calculate home win, away win and draw probabilities.

        Given a home team and away team (or lists thereof), calculate the probabilites
        of the overall results (home win, away win, draw).

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.
            knockout : If True only consider the probability of wins (exclude draws).
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            Dict[str, Union[float, np.ndarray]]: A dictionary with keys "home_win",
                "draw" and "away_win". Values are probabilities of each outcome.
        """
        (
            home_team,
            away_team,
            neutral_venue,
        ) = self._parse_fixture_args(home_team, away_team, neutral_venue)
        # compute probabilities for all scorelines
        probs, home_goals, away_goals = self.predict_score_grid_proba(
            home_team, away_team, neutral_venue, max_goals=max_goals
        )
        # obtain outcome probabilities by summing the appropriate elements of the grid
        home_win = probs[:, home_goals > away_goals].sum(axis=-1)
        draw = probs[:, home_goals == away_goals].sum(axis=-1)
        away_win = probs[:, home_goals < away_goals].sum(axis=-1)

        if knockout:
            # don't consider draws (renormalise with home win and away win only)
            norm = home_win + away_win
            return {"home_win": home_win / norm, "away_win": away_win / norm}

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }

    def sample_score(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
        num_samples: int = 1,
        random_state: int = None,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> Dict[str, jnp.array]:
        """Sample scoreline between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.
            num_samples (int, optional): number of simulations. Defaults to 1.
            random_state (int, optional): seed. Defaults to None.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            Dict[str, Union[float, np.ndarray]]: A dictionary with keys "home_score" and
                "away_score". Values are the simulated goals scored in each simulation.
        """
        (
            home_team,
            away_team,
            neutral_venue,
        ) = self._parse_fixture_args(home_team, away_team, neutral_venue)
        if random_state is None:
            random_state = int(datetime.now().timestamp() * 100)

        probs, home_goals, away_goals = self.predict_score_grid_proba(
            home_team, away_team, neutral_venue, max_goals=max_goals
        )

        home_goals = jnp.array(home_goals.flatten(), DTYPES["goals"])
        away_goals = jnp.array(away_goals.flatten(), DTYPES["goals"])

        rng_key = jax.random.PRNGKey(random_state)
        sample_idx = map_choice(
            rng_key,
            jnp.arange(len(home_goals), dtype="uint32"),
            num_samples,
            probs.reshape((len(home_team), -1)),
        )
        sample_scores_home = home_goals[sample_idx]
        sample_scores_away = away_goals[sample_idx]

        return {"home_score": sample_scores_home, "away_score": sample_scores_away}

    def sample_outcome(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
        knockout: bool = False,
        num_samples: int = 1,
        random_state: int = None,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> np.array:
        """Sample outcome of match between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.
            knockout : If True only consider the probability of wins (exclude draws).
            num_samples (int, optional): number of simulations. Defaults to 1.
            random_state (int, optional): seed. Defaults to None.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            np.array: Array of strings representing the winning team or 'Draw'
                ('Draw' only considered if knockout is False).
        """
        (
            home_team,
            away_team,
            neutral_venue,
        ) = self._parse_fixture_args(home_team, away_team, neutral_venue)

        if random_state is None:
            random_state = int(datetime.now().timestamp() * 100)

        probs = self.predict_outcome_proba(
            home_team, away_team, neutral_venue, knockout, max_goals=max_goals
        )
        if knockout:
            probs = jnp.array([probs["home_win"], probs["away_win"]]).T
        else:
            probs = jnp.array([probs["home_win"], probs["draw"], probs["away_win"]]).T

        rng_key = jax.random.PRNGKey(random_state)
        sample_idx = map_choice(
            rng_key,
            jnp.arange(probs.shape[1], dtype="uint32"),
            num_samples,
            probs,
        )

        winner = np.empty((len(home_team), num_samples), dtype=DTYPES["teams"])
        home_team_rep = home_team.repeat(num_samples).reshape(
            (len(home_team), num_samples)
        )
        away_team_rep = away_team.repeat(num_samples).reshape(
            (len(home_team), num_samples)
        )
        winner[sample_idx == 0] = home_team_rep[sample_idx == 0]
        if knockout:
            winner[sample_idx == 1] = away_team_rep[sample_idx == 1]
        else:
            winner[sample_idx == 2] = away_team_rep[sample_idx == 2]
            winner[sample_idx == 1] = len(self.teams)  # Temporary index for 'Draw'

        _teams_with_draw = np.append(self.teams, "Draw")
        return _teams_with_draw[winner]

    def predict_score_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        home: Optional[bool] = True,
        neutral_venue: Optional[int] = 0,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> jnp.array:
        """
        Compute the probability that a team will score n goals.
        Given a team and an opponent, calculate the probability that the team will
        score n goals against this opponent.

        Args:
            n (Union[int, Iterable[int]]): number of goals scored.
            team (Union[str, Iterable[str]]): name of the team scoring the goals.
            opponent (Union[str, Iterable[str]]): name of the opponent.
            home (Optional[bool]): whether team is at home.
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            jnp.array: Probability that team scores n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n
        (
            team,
            opponent,
            _,
        ) = self._parse_fixture_args(team, opponent, neutral_venue)
        # flat lists of all possible scorelines with team scoring n goals
        team_rep = np.repeat(team, (max_goals + 1) * len(n))
        opponent_rep = np.repeat(opponent, (max_goals + 1) * len(n))
        n_rep = np.resize(n, (max_goals + 1) * len(n))
        x_rep = np.repeat(np.arange(max_goals + 1), len(n))
        neutral_venue_rep = np.repeat(neutral_venue, (max_goals + 1) * len(n))

        probs = (
            self.predict_score_proba(
                team_rep,
                opponent_rep,
                n_rep,
                x_rep,
                neutral_venue_rep,
            )
            if home
            else self.predict_score_proba(
                opponent_rep,
                team_rep,
                x_rep,
                n_rep,
                neutral_venue_rep,
            )
        ).reshape(max_goals + 1, len(n))

        # sum probability of all scorelines where team scored n goals
        return probs.sum(axis=0)

    def predict_concede_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        home: Optional[bool] = True,
        neutral_venue: Optional[int] = 0,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> jnp.array:
        """
        Compute the probability that a team will concede n goals.
        Given a team and an opponent, calculate the probability that the team will
        concede n goals against this opponent.

        Args:
            n (Union[int, Iterable[int]]): number of goals conceded.
            team (Union[str, Iterable[str]]): name of the team conceding the goals.
            opponent (Union[str, Iterable[str]]): name of the opponent.
            home (Optional[bool]): whether team is at home.
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
                else 0.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            jnp.array: Probability that team concedes n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n
        (
            team,
            opponent,
            _,
        ) = self._parse_fixture_args(team, opponent, neutral_venue)
        # flat lists of all possible scorelines with team conceding n goals
        team_rep = np.repeat(team, (max_goals + 1) * len(n))
        opponent_rep = np.repeat(opponent, (max_goals + 1) * len(n))
        n_rep = np.resize(n, (max_goals + 1) * len(n))
        x_rep = np.repeat(np.arange(max_goals + 1), len(n))
        neutral_venue_rep = np.repeat(neutral_venue, (max_goals + 1) * len(n))

        probs = (
            self.predict_score_proba(
                team_rep,
                opponent_rep,
                x_rep,
                n_rep,
                neutral_venue_rep,
            )
            if home
            else self.predict_score_proba(
                opponent_rep,
                team_rep,
                n_rep,
                x_rep,
                neutral_venue_rep,
            )
        ).reshape(max_goals + 1, len(n))

        # sum probability all scorelines where team conceded n goals
        return probs.sum(axis=0)

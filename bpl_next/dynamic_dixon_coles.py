"""Implementation of the neutral model with dynamic parameters in the current version of bpl."""

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

from bpl_next._util import compute_corr_coef_bounds, dixon_coles_correlation_term
from bpl_next.base import MAX_GOALS

__all__ = ["DynamicNeutralDixonColesMatchPredictor"]


class DynamicNeutralDixonColesMatchPredictor:
    """
    A Dixon-Coles like model for predicting match outcomes, modified to:
    - Work for matches in neutral venues (e.g. international tournaments)
    - Add separate home & away, defence & attack, advantages/disadvantages for each team
    """

    # pylint: disable=duplicate-code
    def __init__(self):
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_attack = None
        self.away_attack = None
        self.home_defence = None
        self.away_defence = None
        self.corr_coef = None
        self.u = None
        self.rho = None
        self.attack_coefficients = None
        self.defence_coefficients = None
        self.mean_attack = None
        self.mean_defence = None
        self.std_defence = None
        self.std_attack = None
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
        self._team_covariates_mean = None
        self._team_covariates_std = None

    # pylint: disable=too-many-locals,duplicate-code
    @staticmethod
    def _model(
        home_team: jnp.array,
        away_team: jnp.array,
        gameweek: jnp.array,
        num_teams: int,
        num_gameweeks: int,
        home_goals: Iterable[int],
        away_goals: Iterable[int],
        neutral_venue: Iterable[int],
        team_covariates: Optional[np.array],
    ):
        with numpyro.plate("gameweek", num_gameweeks):
            mean_home_attack = numpyro.sample("mean_home_attack", dist.Normal(0.1, 0.2))
            mean_away_attack = numpyro.sample(
                "mean_away_attack", dist.Normal(-0.1, 0.2)
            )
            mean_home_defence = numpyro.sample(
                "mean_home_defence", dist.Normal(0.1, 0.2)
            )
            mean_away_defence = numpyro.sample(
                "mean_away_defence", dist.Normal(-0.1, 0.2)
            )
            std_home_attack = numpyro.sample(
                "std_home_attack", dist.HalfNormal(scale=1.0)
            )
            std_away_attack = numpyro.sample(
                "std_away_attack", dist.HalfNormal(scale=1.0)
            )
            std_home_defence = numpyro.sample(
                "std_home_defence", dist.HalfNormal(scale=1.0)
            )
            std_away_defence = numpyro.sample(
                "std_away_defence", dist.HalfNormal(scale=1.0)
            )
            std_attack = numpyro.sample("std_attack", dist.HalfNormal(scale=1.0))
            std_defence = numpyro.sample("std_defence", dist.HalfNormal(scale=1.0))
        mean_attack = 0.0
        mean_defence = numpyro.sample("mean_defence", dist.Normal(loc=0.0, scale=1.0))

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

        with numpyro.plate("teams", num_teams):
            with numpyro.plate("gameweek", num_gameweeks):
                u = numpyro.sample(
                    "u", dist.Beta(concentration1=2.0, concentration0=4.0)
                )
                rho = numpyro.deterministic("rho", 2.0 * u - 1.0)
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
                        dist.Normal(
                            jnp.repeat(mean_home_attack, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                            jnp.repeat(std_home_attack, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                        ),
                    )
                with reparam(config={"away_attack": LocScaleReparam(centered=0)}):
                    away_attack = numpyro.sample(
                        "away_attack",
                        dist.Normal(
                            jnp.repeat(mean_away_attack, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                            jnp.repeat(std_away_attack, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                        ),
                    )
                with reparam(config={"home_defence": LocScaleReparam(centered=0)}):
                    home_defence = numpyro.sample(
                        "home_defence",
                        dist.Normal(
                            jnp.repeat(mean_home_defence, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                            jnp.repeat(std_home_defence, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                        ),
                    )
                with reparam(config={"away_defence": LocScaleReparam(centered=0)}):
                    away_defence = numpyro.sample(
                        "away_defence",
                        dist.Normal(
                            jnp.repeat(mean_away_defence, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                            jnp.repeat(std_away_defence, num_teams).reshape(
                                num_gameweeks, num_teams
                            ),
                        ),
                    )
        # why aren't these standardised? if they are, how?
        # what is standardised_attack and standardised_defence actually doing
        attack = jnp.empty([num_gameweeks, num_teams])
        defence = jnp.empty([num_gameweeks, num_teams])
        attack.at[0, ...].set(
            numpyro.deterministic(
                "attack_0",
                attack_prior_mean + standardised_attack[0, ...] * std_attack[0],
            )
        )
        defence.at[0, ...].set(
            numpyro.deterministic(
                "defence_0",
                defence_prior_mean + standardised_defence[0, ...] * std_defence[0],
            )
        )
        for j in range(1, num_gameweeks):
            attack.at[j, ...].set(
                numpyro.deterministic(
                    f"attack_{j}",
                    attack[j - 1, ...] + standardised_attack[j, ...] * std_attack[j],
                )
            )
            defence.at[j, ...].set(
                numpyro.deterministic(
                    f"defence_{j}",
                    defence[j - 1, ...] + standardised_defence[j, ...] * std_defence[j],
                )
            )

        expected_home_goals = jnp.exp(
            attack[gameweek, home_team]
            - defence[gameweek, away_team]
            + (1 - neutral_venue) * home_attack[gameweek, home_team]
            - (1 - neutral_venue) * away_defence[gameweek, away_team]
        )
        expected_away_goals = jnp.exp(
            attack[gameweek, away_team]
            - defence[gameweek, home_team]
            + (1 - neutral_venue) * away_attack[gameweek, away_team]
            - (1 - neutral_venue) * home_defence[gameweek, home_team]
        )

        numpyro.sample(
            "home_goals", dist.Poisson(expected_home_goals).to_event(1), obs=home_goals
        )
        numpyro.sample(
            "away_goals", dist.Poisson(expected_away_goals).to_event(1), obs=away_goals
        )

        # impose bounds on the correlation coefficient
        corr_coef_raw = numpyro.sample("corr_coef_raw", dist.Uniform(low=0.0, high=1.0))
        LB, UB = compute_corr_coef_bounds(expected_home_goals, expected_away_goals)
        corr_coef = numpyro.deterministic("corr_coef", LB + corr_coef_raw * (UB - LB))
        corr_term = dixon_coles_correlation_term(
            home_goals, away_goals, expected_home_goals, expected_away_goals, corr_coef
        )
        numpyro.factor("correlation_term", corr_term.sum(axis=-1))

    # pylint: disable=arguments-differ,too-many-arguments,duplicate-code
    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        random_state: int = 42,
        num_warmup: int = 500,
        num_samples: int = 1000,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DynamicNeutralDixonColesMatchPredictor:
        """
        Fit the model.
        """
        home_team = training_data["home_team"]
        away_team = training_data["away_team"]
        team_covariates = training_data.get("team_covariates")

        self.teams = sorted(list(set(home_team) | set(away_team)))
        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        if team_covariates:
            if set(team_covariates.keys()) != set(self.teams):
                raise ValueError(
                    "team_covariates must contain all the teams in the data."
                )
            team_covariates = jnp.array([team_covariates[t] for t in self.teams])
            self._team_covariates_mean = team_covariates.mean(axis=0)
            self._team_covariates_std = team_covariates.std(axis=0)

        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            **(mcmc_kwargs or {}),
        )
        rng_key = jax.random.PRNGKey(random_state)
        num_gameweeks = max(np.array(training_data["gameweek"], dtype=int))
        mcmc.run(
            rng_key,
            home_ind,
            away_ind,
            np.array(training_data["gameweek"], dtype=int),
            len(self.teams),
            num_gameweeks,
            np.array(training_data["home_goals"]),
            np.array(training_data["away_goals"]),
            np.array(training_data["neutral_venue"]),
            team_covariates=team_covariates,
            **(run_kwargs or {}),
        )

        samples = mcmc.get_samples()
        print(samples["attack_0"].shape)
        print(samples["attack_1"].shape)
        print(samples[f"attack_{num_gameweeks-1}"].shape)
        print([samples[f"attack_{j}"].shape for j in range(num_gameweeks)])
        print([samples[f"defence_{j}"].shape for j in range(num_gameweeks)])
        self.attack = [samples[f"attack_{j}"] for j in range(num_gameweeks)]
        self.defence = [samples[f"defence_{j}"] for j in range(num_gameweeks)]
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
        self.mean_away_attack = samples["mean_home_attack"]
        self.mean_home_defence = samples["mean_home_defence"]
        self.mean_away_defence = samples["mean_away_defence"]
        self.std_home_attack = samples["std_home_attack"]
        self.std_away_attack = samples["std_away_attack"]
        self.std_home_defence = samples["std_home_defence"]
        self.std_away_defence = samples["std_away_defence"]
        self.standardised_attack = samples["standardised_attack"]
        self.standardised_defence = samples["standardised_defence"]

        return self

    def _calculate_expected_goals(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
    ) -> Tuple[jnp.array, jnp.array]:
        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])
        neutral_venue = jnp.array(neutral_venue)

        attack_home, defence_home = self.attack[:, home_ind], self.defence[:, home_ind]
        attack_away, defence_away = self.attack[:, away_ind], self.defence[:, away_ind]

        home_rate = jnp.exp(
            attack_home
            - defence_away
            + (1 - neutral_venue) * self.home_attack[:, home_ind]
            + (1 - neutral_venue) * self.away_defence[:, away_ind]
        )
        away_rate = jnp.exp(
            attack_away
            - defence_home
            - (1 - neutral_venue) * self.away_attack[:, away_ind]
            - (1 - neutral_venue) * self.home_defence[:, home_ind]
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
        """
        Predict probabilities for scorelines.
        """

        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

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

        home_probs = jnp.exp(dist.Poisson(expected_home_goals).log_prob(home_goals))
        away_probs = jnp.exp(dist.Poisson(expected_away_goals).log_prob(away_goals))

        sampled_probs = jnp.exp(corr_term) * home_probs * away_probs
        return sampled_probs.mean(axis=0)

    def add_new_team(self, team_name: str, team_covariates: Optional[np.array] = None):
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
        home_attack = np.random.normal(loc=self.mean_home, scale=self.std_home)
        away_attack = np.random.normal(loc=self.mean_home, scale=self.std_home)
        home_defence = np.random.normal(loc=self.mean_home, scale=self.std_home)
        away_defence = np.random.normal(loc=self.mean_home, scale=self.std_home)
        attack = mean_attack + log_a_tilde * self.std_attack
        defence = mean_defence + log_b_tilde * self.std_defence

        self.teams.append(team_name)
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

    def predict_outcome_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
    ) -> Dict[str, jnp.array]:
        """Calculate home win, away win and draw probabilities.

        Given a home team and away team (or lists thereof), calculate the probabilites
        of the overall results (home win, away win, draw).

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            neutral_venue (Union[int, Iterable[int]]): 1 if game played at neutral venue,
            else 0

        Returns:
            Dict[str, Union[float, np.ndarray]]: A dictionary with keys "home_win",
                "away_win" and "draw". Values are probabilities of each outcome.
        """
        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

        # make a grid of scorelines up to plausible limits
        n_goals = np.arange(0, MAX_GOALS + 1)
        x, y = np.meshgrid(n_goals, n_goals, indexing="ij")
        x_flat = jnp.tile(x.reshape((MAX_GOALS + 1) ** 2), len(home_team))
        y_flat = jnp.tile(y.reshape((MAX_GOALS + 1) ** 2), len(home_team))
        home_team_rep = np.repeat(home_team, (MAX_GOALS + 1) ** 2)
        away_team_rep = np.repeat(away_team, (MAX_GOALS + 1) ** 2)
        neutral_venue_rep = np.repeat(neutral_venue, (MAX_GOALS + 1) ** 2)

        # evaluate the probability of scorelines at each gridpoint
        probs = self.predict_score_proba(
            home_team_rep, away_team_rep, x_flat, y_flat, neutral_venue_rep
        ).reshape(len(home_team), MAX_GOALS + 1, MAX_GOALS + 1)

        # obtain outcome probabilities by summing the appropriate elements of the grid
        prob_home_win = probs[:, x > y].sum(axis=-1)
        prob_away_win = probs[:, x < y].sum(axis=-1)
        prob_draw = probs[:, x == y].sum(axis=-1)

        return {"home_win": prob_home_win, "away_win": prob_away_win, "draw": prob_draw}

    def predict_score_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        home: Optional[bool] = True,
        neutral_venue: Optional[int] = 0,
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
            else 0

        Returns:
            jnp.array: Probability that team scores n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n

        # flat lists of all possible scorelines with team scoring n goals
        team_rep = np.repeat(team, (MAX_GOALS + 1) * len(n))
        opponent_rep = np.repeat(opponent, (MAX_GOALS + 1) * len(n))
        n_rep = np.resize(n, (MAX_GOALS + 1) * len(n))
        x_rep = np.repeat(np.arange(MAX_GOALS + 1), len(n))
        neutral_venue_rep = np.repeat(neutral_venue, (MAX_GOALS + 1) * len(n))

        probs = (
            self.predict_score_proba(
                team_rep, opponent_rep, n_rep, x_rep, neutral_venue_rep
            )
            if home
            else self.predict_score_proba(
                opponent_rep, team_rep, x_rep, n_rep, neutral_venue_rep
            )
        ).reshape(MAX_GOALS + 1, len(n))

        # sum probability of all scorelines where team scored n goals
        return probs.sum(axis=0)

    def predict_concede_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        home: Optional[bool] = True,
        neutral_venue: Optional[int] = 0,
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
            else 0

        Returns:
            jnp.array: Probability that team concedes n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n

        # flat lists of all possible scorelines with team conceding n goals
        team_rep = np.repeat(team, (MAX_GOALS + 1) * len(n))
        opponent_rep = np.repeat(opponent, (MAX_GOALS + 1) * len(n))
        n_rep = np.resize(n, (MAX_GOALS + 1) * len(n))
        x_rep = np.repeat(np.arange(MAX_GOALS + 1), len(n))
        neutral_venue_rep = np.repeat(neutral_venue, (MAX_GOALS + 1) * len(n))

        probs = (
            self.predict_score_proba(
                team_rep, opponent_rep, x_rep, n_rep, neutral_venue_rep
            )
            if home
            else self.predict_score_proba(
                opponent_rep, team_rep, n_rep, x_rep, neutral_venue_rep
            )
        ).reshape(MAX_GOALS + 1, len(n))

        # sum probability all scorelines where team conceded n goals
        return probs.sum(axis=0)

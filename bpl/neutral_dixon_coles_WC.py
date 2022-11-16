"""Implementation of the neutral model in the current version of bpl for predicting the World Cup."""
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

from bpl._util import compute_corr_coef_bounds, dixon_coles_correlation_term
from bpl.base import MAX_GOALS

__all__ = ["NeutralDixonColesMatchPredictorWC"]


def _str_to_list(*args):
    return ([x] if isinstance(x, str) else x for x in args)


# pylint: disable=too-many-instance-attributes
class NeutralDixonColesMatchPredictorWC:
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
        self._team_covariates_mean = None
        self._team_covariates_std = None

    # pylint: disable=too-many-arguments,too-many-locals,duplicate-code
    @staticmethod
    def _model(
        home_team: jnp.array,
        away_team: jnp.array,
        num_teams: int,
        home_conf: jnp.array,
        away_conf: jnp.array,
        num_conferences: int,
        home_goals: Iterable[int],
        away_goals: Iterable[int],
        neutral_venue: Iterable[int],
        time_diff: Iterable[float],
        epsilon: float,
        game_weights: Iterable[float],
        team_covariates: Optional[np.array] = None,
    ):
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
            attack_prior_mean = mean_attack
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
        attack = numpyro.deterministic(
            "attack", attack_prior_mean + standardised_attack * std_attack
        )
        defence = numpyro.deterministic(
            "defence", defence_prior_mean + standardised_defence * std_defence
        )

        with numpyro.plate("confederations", num_conferences):
            with reparam(
                config={"confederation_strength": LocScaleReparam(centered=0)}
            ):
                confederation_strength = numpyro.sample(
                    "confederation_strength", dist.Normal(0.0, 1.0)
                )

        expected_home_goals = jnp.exp(
            attack[home_team]
            - defence[away_team]
            + confederation_strength[home_conf]
            - confederation_strength[away_conf]
            + (1 - neutral_venue) * home_attack[home_team]
            - (1 - neutral_venue) * away_defence[away_team]
        )
        expected_away_goals = jnp.exp(
            attack[away_team]
            - defence[home_team]
            + confederation_strength[away_conf]
            - confederation_strength[home_conf]
            + (1 - neutral_venue) * away_attack[away_team]
            - (1 - neutral_venue) * home_defence[home_team]
        )

        weights = jnp.exp(-epsilon * time_diff) * game_weights
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

    # pylint: disable=arguments-differ,too-many-arguments,duplicate-code
    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        epsilon: float = 0.0,
        random_state: int = 42,
        num_warmup: int = 500,
        num_samples: int = 1000,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NeutralDixonColesMatchPredictorWC:

        home_team = training_data["home_team"]
        away_team = training_data["away_team"]
        team_covariates = training_data.get("team_covariates")
        home_team_conf = training_data["home_conf"]
        away_team_conf = training_data["away_conf"]

        self.teams = sorted(list(set(home_team) | set(away_team)))
        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])

        self.conferences = sorted(list(set(home_team_conf) | set(away_team_conf)))
        # lookup for what each number represents
        self.conferences_ref = dict(zip(range(len(self.conferences)), self.conferences))
        home_conf_ind = jnp.array([self.conferences.index(t) for t in home_team_conf])
        away_conf_ind = jnp.array([self.conferences.index(t) for t in away_team_conf])

        self.time_diff = training_data["time_diff"]
        self.game_weights = training_data["game_weights"]

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
        mcmc.run(
            rng_key,
            home_ind,
            away_ind,
            len(self.teams),
            home_conf_ind,
            away_conf_ind,
            len(self.conferences),
            np.array(training_data["home_goals"]),
            np.array(training_data["away_goals"]),
            np.array(training_data["neutral_venue"]),
            self.time_diff,
            epsilon,
            self.game_weights,
            team_covariates=team_covariates,
            **(run_kwargs or {}),
        )

        samples = mcmc.get_samples()
        self.confederation_strength = samples["confederation_strength"]
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
        self.epsilon = epsilon

        return self

    def _calculate_expected_goals(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_conf: Union[str, Iterable[str]],
        away_conf: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
    ) -> Tuple[jnp.array, jnp.array]:

        home_ind = jnp.array([self.teams.index(t) for t in home_team])
        away_ind = jnp.array([self.teams.index(t) for t in away_team])
        neutral_venue = jnp.array(neutral_venue)
        home_conf_ind = jnp.array([self.conferences.index(t) for t in home_conf])
        away_conf_ind = jnp.array([self.conferences.index(t) for t in away_conf])

        attack_home, defence_home = self.attack[:, home_ind], self.defence[:, home_ind]
        attack_away, defence_away = self.attack[:, away_ind], self.defence[:, away_ind]
        home_conf_strength = self.confederation_strength[:, home_conf_ind]
        away_conf_strength = self.confederation_strength[:, away_conf_ind]

        home_rate = jnp.exp(
            attack_home
            - defence_away
            + home_conf_strength
            - away_conf_strength
            + (1 - neutral_venue) * self.home_attack[:, home_ind]
            - (1 - neutral_venue) * self.away_defence[:, away_ind]
        )
        away_rate = jnp.exp(
            attack_away
            - defence_home
            + away_conf_strength
            - home_conf_strength
            + (1 - neutral_venue) * self.away_attack[:, away_ind]
            - (1 - neutral_venue) * self.home_defence[:, home_ind]
        )

        return home_rate, away_rate

    def predict_score_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_conf: Union[str, Iterable[str]],
        away_conf: Union[str, Iterable[str]],
        home_goals: Union[int, Iterable[int]],
        away_goals: Union[int, Iterable[int]],
        neutral_venue: Union[int, Iterable[int]],
    ) -> jnp.array:
        home_team, away_team, home_conf, away_conf = _str_to_list(
            home_team, away_team, home_conf, away_conf
        )

        expected_home_goals, expected_away_goals = self._calculate_expected_goals(
            home_team, away_team, home_conf, away_conf, neutral_venue
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

    def predict_score_grid_proba(
        self, home_team, away_team, home_conf, away_conf, neutral_venue
    ):
        """Compute probabilities of all plausible scorelines"""
        home_team, away_team, home_conf, away_conf = _str_to_list(
            home_team, away_team, home_conf, away_conf
        )

        n_goals = np.arange(0, MAX_GOALS + 1)
        home_goals, away_goals = np.meshgrid(n_goals, n_goals, indexing="ij")
        home_goals_flat = jnp.tile(
            home_goals.reshape((MAX_GOALS + 1) ** 2), len(home_team)
        )
        away_goals_flat = jnp.tile(
            away_goals.reshape((MAX_GOALS + 1) ** 2), len(home_team)
        )
        home_team_rep = np.repeat(home_team, (MAX_GOALS + 1) ** 2)
        away_team_rep = np.repeat(away_team, (MAX_GOALS + 1) ** 2)
        home_conf_rep = np.repeat(home_conf, (MAX_GOALS + 1) ** 2)
        away_conf_rep = np.repeat(away_conf, (MAX_GOALS + 1) ** 2)
        neutral_venue_rep = np.repeat(neutral_venue, (MAX_GOALS + 1) ** 2)

        probs = self.predict_score_proba(
            home_team_rep,
            away_team_rep,
            home_conf_rep,
            away_conf_rep,
            home_goals_flat,
            away_goals_flat,
            neutral_venue_rep,
        ).reshape(len(home_team), MAX_GOALS + 1, MAX_GOALS + 1)

        return probs, home_goals, away_goals

    def predict_outcome_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_conf: Union[str, Iterable[str]],
        away_conf: Union[str, Iterable[str]],
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
        # compute probabilities for all scorelines
        probs, home_goals, away_goals = self.predict_score_grid_proba(
            home_team, away_team, home_conf, away_conf, neutral_venue
        )

        # obtain outcome probabilities by summing the appropriate elements of the grid
        prob_home_win = probs[:, home_goals > away_goals].sum(axis=-1)
        prob_away_win = probs[:, home_goals < away_goals].sum(axis=-1)
        prob_draw = probs[:, home_goals == away_goals].sum(axis=-1)

        return {"home_win": prob_home_win, "away_win": prob_away_win, "draw": prob_draw}

    def simulate_score(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_conf: Union[str, Iterable[str]],
        away_conf: Union[str, Iterable[str]],
        neutral_venue: Union[int, Iterable[int]],
        num_samples: int = 1,
        random_state: int = None,
    ):
        if random_state is None:
            random_state = int(datetime.now().timestamp())

        home_team, away_team, home_conf, away_conf = _str_to_list(
            home_team, away_team, home_conf, away_conf
        )

        probs, home_goals, away_goals = self.predict_score_grid_proba(
            home_team, away_team, home_conf, away_conf, neutral_venue
        )
        home_goals = home_goals.flatten()
        away_goals = away_goals.flatten()
        home_sim_score = np.full((len(home_team), num_samples), np.nan)
        away_sim_score = np.full((len(home_team), num_samples), np.nan)
        rng_key = jax.random.PRNGKey(random_state)

        for fixture in range(len(home_team)):
            new_key, subkey = jax.random.split(rng_key)
            sim_idx = jax.random.choice(
                subkey,
                (MAX_GOALS + 1) ** 2,
                shape=(num_samples,),
                p=probs[fixture].flatten(),
            )
            rng_key = new_key

            home_sim_score[fixture, :] = home_goals[sim_idx]
            away_sim_score[fixture, :] = away_goals[sim_idx]

        return {"home_score": home_sim_score, "away_score": away_sim_score}

    def predict_score_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        team_conf: Union[str, Iterable[str]],
        opponent_conf: Union[str, Iterable[str]],
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
        team_conf_rep = np.repeat(team_conf, (MAX_GOALS + 1) * len(n))
        opponent_conf_rep = np.repeat(opponent_conf, (MAX_GOALS + 1) * len(n))
        n_rep = np.resize(n, (MAX_GOALS + 1) * len(n))
        x_rep = np.repeat(np.arange(MAX_GOALS + 1), len(n))
        neutral_venue_rep = np.repeat(neutral_venue, (MAX_GOALS + 1) * len(n))

        probs = (
            self.predict_score_proba(
                team_rep,
                opponent_rep,
                team_conf_rep,
                opponent_conf_rep,
                n_rep,
                x_rep,
                neutral_venue_rep,
            )
            if home
            else self.predict_score_proba(
                opponent_rep,
                team_rep,
                opponent_conf_rep,
                team_conf_rep,
                x_rep,
                n_rep,
                neutral_venue_rep,
            )
        ).reshape(MAX_GOALS + 1, len(n))

        # sum probability of all scorelines where team scored n goals
        return probs.sum(axis=0)

    def predict_concede_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        team_conf: Union[str, Iterable[str]],
        opponent_conf: Union[str, Iterable[str]],
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
        team_conf_rep = np.repeat(team_conf, (MAX_GOALS + 1) * len(n))
        opponent_conf_rep = np.repeat(opponent_conf, (MAX_GOALS + 1) * len(n))
        n_rep = np.resize(n, (MAX_GOALS + 1) * len(n))
        x_rep = np.repeat(np.arange(MAX_GOALS + 1), len(n))
        neutral_venue_rep = np.repeat(neutral_venue, (MAX_GOALS + 1) * len(n))

        probs = (
            self.predict_score_proba(
                team_rep,
                opponent_rep,
                team_conf_rep,
                opponent_conf_rep,
                x_rep,
                n_rep,
                neutral_venue_rep,
            )
            if home
            else self.predict_score_proba(
                opponent_rep,
                team_rep,
                opponent_conf_rep,
                team_conf_rep,
                n_rep,
                x_rep,
                neutral_venue_rep,
            )
        ).reshape(MAX_GOALS + 1, len(n))

        # sum probability all scorelines where team conceded n goals
        return probs.sum(axis=0)

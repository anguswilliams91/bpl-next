"""Implementation of the probabilistic model for soccer matches."""
from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Iterable, Optional, Union

import jax.numpy as jnp
import numpy as np

MAX_GOALS = 15
DTYPES = {
    "goals": "uint8",
    "teams": "uint16",
    "conferences": "uint8",
    "venue": "uint8",
    "outcome": "uint8",
}


class BaseMatchPredictor:
    """Abstract class for models of football matches."""

    @abstractmethod
    def fit(
        self, training_data: Dict[str, Union[Iterable[str], Iterable[float]]], **kwargs
    ) -> BaseMatchPredictor:
        """Fit the model to data and return self."""

    @abstractmethod
    def predict_score_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_goals: Union[int, Iterable[int]],
        away_goals: Union[int, Iterable[int]],
    ) -> jnp.array:
        """Return the probability of a particular scoreline.

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

    def predict_outcome_proba(
        self, home_team: Union[str, Iterable[str]], away_team: Union[str, Iterable[str]]
    ) -> Dict[str, jnp.array]:
        """Calculate home win, away win and draw probabilities.

        Given a home team and away team (or lists thereof), calculate the probabilites
        of the overall results (home win, away win, draw).

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).

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

        # evaluate the probability of scorelines at each gridpoint
        probs = self.predict_score_proba(
            home_team_rep, away_team_rep, x_flat, y_flat
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

        Returns:
            jnp.array: Probability that team scores n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n

        # flat lists of all possible scorelines with team scoring n goals
        team_rep = np.repeat(team, (MAX_GOALS + 1) * len(n))
        opponent_rep = np.repeat(opponent, (MAX_GOALS + 1) * len(n))
        n_rep = np.resize(n, (MAX_GOALS + 1) * len(n))
        x_rep = np.repeat(np.arange(MAX_GOALS + 1), len(n))

        probs = (
            self.predict_score_proba(team_rep, opponent_rep, n_rep, x_rep)
            if home
            else self.predict_score_proba(opponent_rep, team_rep, x_rep, n_rep)
        ).reshape(MAX_GOALS + 1, len(n))

        # sum probability of all scorelines where team scored n goals
        return probs.sum(axis=0)

    def predict_concede_n_proba(
        self,
        n: Union[int, Iterable[int]],
        team: Union[str, Iterable[str]],
        opponent: Union[str, Iterable[str]],
        home: Optional[bool] = True,
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

        Returns:
            jnp.array: Probability that team concedes n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n

        # flat lists of all possible scorelines with team conceding n goals
        team_rep = np.repeat(team, (MAX_GOALS + 1) * len(n))
        opponent_rep = np.repeat(opponent, (MAX_GOALS + 1) * len(n))
        n_rep = np.resize(n, (MAX_GOALS + 1) * len(n))
        x_rep = np.repeat(np.arange(MAX_GOALS + 1), len(n))

        probs = (
            self.predict_score_proba(team_rep, opponent_rep, x_rep, n_rep)
            if home
            else self.predict_score_proba(opponent_rep, team_rep, n_rep, x_rep)
        ).reshape(MAX_GOALS + 1, len(n))

        # sum probability all scorelines where team conceded n goals
        return probs.sum(axis=0)

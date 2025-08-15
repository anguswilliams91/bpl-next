"""Implementation of the probabilistic model for soccer matches."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from bpl_next._util import map_choice, str_to_list

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

    def __init__(self):
        # list of all unique team names used to create integer indicator for each team
        # and dict mapping from the team names to their corresponding integer indicator
        self.teams = None
        self._teams_dict = None

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

    def _parse_fixture_args(self, home_team, away_team):
        home_team, away_team = str_to_list(home_team, away_team)
        if isinstance(home_team[0], str):
            home_team = jnp.array(
                [self._teams_dict[t] for t in home_team], DTYPES["teams"]
            )
        if isinstance(away_team[0], str):
            away_team = jnp.array(
                [self._teams_dict[t] for t in away_team], DTYPES["teams"]
            )
        return home_team, away_team

    def predict_score_grid_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        max_goals: Optional[int] = MAX_GOALS,
    ) -> Tuple[jnp.array, np.array, np.array]:
        """Calculate scoreline probabilities between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            Tuple[jnp.array, np.array, np.array]: Tuple of the following grids (as arrays):
                probability of scorelines, home goals and away goals scored grids
        """
        home_team, away_team = self._parse_fixture_args(home_team, away_team)

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

        probs = self.predict_score_proba(
            home_team_rep,
            away_team_rep,
            home_goals_flat,
            away_goals_flat,
        ).reshape(len(home_team), max_goals + 1, max_goals + 1)
        return probs, home_goals, away_goals

    def predict_outcome_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        max_goals: Optional[int] = MAX_GOALS,
    ) -> Dict[str, jnp.array]:
        """Calculate home win, away win and draw probabilities.

        Given a home team and away team (or lists thereof), calculate the probabilites
        of the overall results (home win, away win, draw).

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            Dict[str, Union[float, np.ndarray]]: A dictionary with keys "home_win",
                "draw" and "away_win". Values are probabilities of each outcome.
        """
        home_team, away_team = self._parse_fixture_args(home_team, away_team)
        # compute probabilities for all scorelines
        probs, home_goals, away_goals = self.predict_score_grid_proba(
            home_team, away_team, max_goals=max_goals
        )
        # obtain outcome probabilities by summing the appropriate elements of the grid
        home_win = probs[:, home_goals > away_goals].sum(axis=-1)
        draw = probs[:, home_goals == away_goals].sum(axis=-1)
        away_win = probs[:, home_goals < away_goals].sum(axis=-1)

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }

    def sample_score(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        num_samples: int = 1,
        random_state: int = None,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> Dict[str, jnp.array]:
        """Sample scoreline between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            num_samples (int, optional): number of simulations. Defaults to 1.
            random_state (int, optional): seed. Defaults to None.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            Dict[str, Union[float, np.ndarray]]: A dictionary with keys "home_score" and
                "away_score". Values are the simulated goals scored in each simulation.
        """
        home_team, away_team = self._parse_fixture_args(home_team, away_team)
        if random_state is None:
            random_state = int(datetime.now().timestamp() * 100)

        probs, home_goals, away_goals = self.predict_score_grid_proba(
            home_team,
            away_team,
            max_goals=max_goals,
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
        num_samples: int = 1,
        random_state: int = None,
        max_goals: Optional[int] = MAX_GOALS,
    ) -> np.array:
        """Sample outcome of match between two teams.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            num_samples (int, optional): number of simulations. Defaults to 1.
            random_state (int, optional): seed. Defaults to None.
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            np.array: Array of strings representing the winning team or 'Draw'
        """
        home_team, away_team = self._parse_fixture_args(home_team, away_team)

        if random_state is None:
            random_state = int(datetime.now().timestamp() * 100)

        probs = self.predict_outcome_proba(home_team, away_team, max_goals=max_goals)
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
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            jnp.array: Probability that team scores n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n
        team, opponent = self._parse_fixture_args(team, opponent)
        # flat lists of all possible scorelines with team scoring n goals
        team_rep = np.repeat(team, (max_goals + 1) * len(n))
        opponent_rep = np.repeat(opponent, (max_goals + 1) * len(n))
        n_rep = np.resize(n, (max_goals + 1) * len(n))
        x_rep = np.repeat(np.arange(max_goals + 1), len(n))

        probs = (
            self.predict_score_proba(
                team_rep,
                opponent_rep,
                n_rep,
                x_rep,
            )
            if home
            else self.predict_score_proba(
                opponent_rep,
                team_rep,
                x_rep,
                n_rep,
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
            max_goals (Optional[int]): Compute scorelines where each team scores up to
                this many goals. Defaults to bpl.base.MAX_GOALS.

        Returns:
            jnp.array: Probability that team concedes n goals against opponent.
        """
        n = [n] if isinstance(n, int) else n
        team, opponent = self._parse_fixture_args(team, opponent)
        # flat lists of all possible scorelines with team conceding n goals
        team_rep = np.repeat(team, (max_goals + 1) * len(n))
        opponent_rep = np.repeat(opponent, (max_goals + 1) * len(n))
        n_rep = np.resize(n, (max_goals + 1) * len(n))
        x_rep = np.repeat(np.arange(max_goals + 1), len(n))

        probs = (
            self.predict_score_proba(
                team_rep,
                opponent_rep,
                x_rep,
                n_rep,
            )
            if home
            else self.predict_score_proba(
                opponent_rep,
                team_rep,
                n_rep,
                x_rep,
            )
        ).reshape(max_goals + 1, len(n))

        # sum probability all scorelines where team conceded n goals
        return probs.sum(axis=0)

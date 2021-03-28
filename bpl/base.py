"""Implementation of the probabilistic model for soccer matches."""
from __future__ import annotations

from abc import abstractmethod, abstractproperty
from typing import Dict, Iterable, Union

import numpy as np


class BaseMatchPredictor:
    """Abstract class for models of football matches."""

    @abstractmethod
    def predict_outcome_proba(
        self, home_team: Union[str, Iterable[str]], away_team: Union[str, Iterable[str]]
    ) -> Dict[str, Union[float, np.ndarray]]:
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
        pass

    @abstractmethod
    def predict_score_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_goals: Union[float, Iterable[float]],
        away_goals: Union[float, Iterable[float]],
    ) -> Union[float, np.ndarray]:
        """Return the probability of a particular scoreline.

        Args:
            home_team (Union[str, Iterable[str]]): name of the home team(s).
            away_team (Union[str, Iterable[str]]): name of the away team(s).
            home_goals (Union[float, Iterable[float]]): number of goals scored by
                the home team(s).
            away_goals (Union[float, Iterable[float]]): number of goals scored by
                the away team(s).

        Returns:
            float: the probability of the given outcome.
        """
        pass

    @abstractmethod
    def fit(
        self, training_data: Dict[str, Union[Iterable[str], Iterable[float]]]
    ) -> BaseMatchPredictor:
        pass

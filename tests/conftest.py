import numpy as np
import pytest


@pytest.fixture
def dummy_data():
    home_team = np.tile(["A", "B", "C", "D"], 20)
    away_team = np.tile(["D", "A", "B", "C"], 20)
    home_goals = np.tile(np.array([3, 0, 1, 2]), 20)
    away_goals = np.tile(np.array([0, 2, 1, 1]), 20)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }

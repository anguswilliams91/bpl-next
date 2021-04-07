import itertools

import numpy as np
import pytest


@pytest.fixture
def dummy_data():

    np.random.seed(42)
    home_mean = 2.1
    away_mean = 1.7

    home_goals = np.random.poisson(home_mean, size=190)
    away_goals = np.random.poisson(away_mean, size=190)

    teams = [str(i) for i in range(20)]
    matchups = itertools.combinations(teams, 2)
    home_team = []
    away_team = []
    for a, b in matchups:
        home_team.append(a)
        away_team.append(b)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }

import itertools

import numpy as np
import pytest


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    home_mean = 2.1
    away_mean = 1.7

    home_goals = np.random.poisson(home_mean, size=380)
    away_goals = np.random.poisson(away_mean, size=380)

    teams = [str(i) for i in range(20)]
    matchups = itertools.permutations(teams, 2)
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


@pytest.fixture
def timed_dummy_data():
    """
    Generates dummy data, including time_diff, for two teams playing each other
    repeatedly where team A wins the first third of the matches, the second third are
    draws, and team B wins the last third (with no home advantage).
    """
    # phases are: team A wins, draws, team B wins
    matches_per_phase = 20
    # alternate home and away team (with no home advantage)
    home_team = ["A", "B"] * int(matches_per_phase / 2) * 3
    away_team = ["B", "A"] * int(matches_per_phase / 2) * 3
    # fill 2-0 / 1-1 / 0-2 results according to phase
    home_goals = (
        [2, 0] * int(matches_per_phase / 2)
        + [1] * matches_per_phase
        + [0, 2] * int(matches_per_phase / 2)
    )
    away_goals = (
        [0, 2] * int(matches_per_phase / 2)
        + [1] * matches_per_phase
        + [2, 0] * int(matches_per_phase / 2)
    )
    time_diff = np.linspace(5, 0, num=matches_per_phase * 3)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "time_diff": time_diff,
    }


@pytest.fixture
def neutral_dummy_data():
    """
    Generate data for 20 teams that play each other home and away,
    (i.e. 380 "league" matches) and then play each other once in neutral venues
    (another 190 "cup" matches).
    For the league matches, have time_diff and game_weights constant,
    while these vary for the neutral "cup" games.
    """
    np.random.seed(42)
    home_mean = 2.1
    neutral_mean = 1.9
    away_mean = 1.7
    neutral_venue = np.array([0] * 380 + [1] * 190)
    home_means = [home_mean if venue == 0 else neutral_mean for venue in neutral_venue]
    away_means = [away_mean if venue == 0 else neutral_mean for venue in neutral_venue]
    home_goals = np.random.poisson(home_means)
    away_goals = np.random.poisson(away_means)
    time_diff_league = np.array([1.0] * 380)
    time_diff_cup = np.linspace(0, 10, num=190)
    time_diff = np.concatenate([time_diff_league, time_diff_cup])
    game_weights_league = np.array([1.0] * 380)
    game_weights_cup = np.random.uniform(0, 10, size=190)
    game_weights = np.concatenate([game_weights_league, game_weights_cup])

    teams = [str(i) for i in range(20)]
    league_matchups = itertools.permutations(teams, 2)
    home_team = []
    away_team = []
    for a, b in league_matchups:
        home_team.append(a)
        away_team.append(b)
    cup_matchups = itertools.combinations(teams, 2)
    for a, b in cup_matchups:
        home_team.append(a)
        away_team.append(b)

    # deterministic assignment of teams to conferences
    home_conf = [str(int(ht) // 4) for ht in home_team]
    away_conf = [str(int(at) // 4) for at in away_team]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_conf": home_conf,
        "away_conf": away_conf,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "neutral_venue": neutral_venue,
        "time_diff": time_diff,
        "game_weights": game_weights,
    }

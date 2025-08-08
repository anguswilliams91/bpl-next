"""Private utility functions."""

from typing import Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


def str_to_list(*args):
    """
    convert all elements of a list into strings.
    """
    return ([x] if isinstance(x, str) else x for x in args)


def compute_corr_coef_bounds(
    expected_home_goals: jnp.array, expected_away_goals: jnp.array
) -> Tuple[float, float]:
    """
    Computes the bounds of the correlation coefficient from dixon and coles paper
    """
    UB = jnp.min(
        jnp.array([jnp.min(1.0 / (expected_home_goals * expected_away_goals)), 1])
    )
    LB = jnp.max(
        jnp.array(
            [jnp.max(-1.0 / expected_home_goals), jnp.max(-1.0 / expected_away_goals)]
        )
    )
    return LB, UB


# pylint: disable=too-many-arguments
def dixon_coles_correlation_term(
    home_goals: Union[int, Iterable[int]],
    away_goals: Union[int, Iterable[int]],
    home_rate: jnp.array,
    away_rate: jnp.array,
    corr_coef: jnp.array,
    weights: Optional[jnp.array] = None,
    tol: Optional[float] = 0,  # workaround to clip negative values to tol to avoid NaNs
) -> jnp.array:
    """
    Calculate correlation term from dixon and coles paper
    """
    if isinstance(home_goals, int):
        home_goals = np.array(home_goals).reshape((1,))
    if isinstance(away_goals, int):
        away_goals = np.array(away_goals).reshape((1,))
    if weights is None:
        weights = jnp.ones(len(home_goals))

    corr_term = jnp.zeros_like(home_rate)
    if weights is None:
        weights = jnp.ones_like(corr_term)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    corr_term = corr_term.at[..., nil_nil].set(
        weights[..., nil_nil]
        * jnp.log(
            jnp.clip(
                1.0
                - corr_coef[..., None]
                * home_rate[..., nil_nil]
                * away_rate[..., nil_nil],
                a_min=tol,
            )
        )
    )

    one_nil = (home_goals == 1) & (away_goals == 0)
    corr_term = corr_term.at[..., one_nil].set(
        weights[..., one_nil]
        * jnp.log(
            jnp.clip(1.0 + corr_coef[..., None] * away_rate[..., one_nil], a_min=tol)
        )
    )
    nil_one = (home_goals == 0) & (away_goals == 1)
    corr_term = corr_term.at[..., nil_one].set(
        weights[..., nil_one]
        * jnp.log(
            jnp.clip(1.0 + corr_coef[..., None] * home_rate[..., nil_one], a_min=tol)
        ),
    )

    one_one = (home_goals == 1) & (away_goals == 1)
    corr_term = corr_term.at[..., one_one].set(
        weights[..., one_one]
        * jnp.log(jnp.clip(1.0 - corr_coef[..., None], a_min=tol)),
    )

    return corr_term


def map_choice(key, a, num_samples, p):
    """
    Map choices.
    """

    def _map_choice_once(probs_and_key):
        probs, rng_key = probs_and_key
        choices = jax.random.choice(
            rng_key,
            a,
            shape=(num_samples,),
            p=probs,
        )
        return choices

    new_keys = jax.random.split(key, p.shape[0])
    return jax.vmap(_map_choice_once)((p, new_keys))


def parse_teams(
    home_team: Iterable[str], away_team: Iterable[str], dtype: str
) -> Tuple[np.ndarray, dict, jnp.ndarray, jnp.ndarray]:
    """Parse home and away teams for a number of fixtures to extract unique names,
    a mapping between team names and indices, and the corresponding indices for each
    fixture.

    Args:
        home_team: Home team for each fixture
        away_team: Away team for each fixture
        dtype (str): Data type to use for team indices

    Returns:
        Unique team names, mapping from team names to indices, home team index for each
            fixture, away team index for each fixture
    """
    teams = np.array(sorted(set(home_team) | set(away_team)))
    teams_dict = {t: i for i, t in enumerate(teams)}
    home_ind = jnp.array([teams_dict[t] for t in home_team], dtype)
    away_ind = jnp.array([teams_dict[t] for t in away_team], dtype)
    return teams, teams_dict, home_ind, away_ind

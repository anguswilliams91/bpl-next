"""Private utility functions."""
from typing import Iterable, Union, Tuple

import jax.numpy as jnp
import numpy as np

def compute_corr_coef_bounds(
    expected_home_goals: jnp.array, expected_away_goals: jnp.array
) -> Tuple[float, float]:
    # computes the bounds of the correlation coefficient from dixon and coles paper
    UB = jnp.min(
        jnp.array([jnp.min(1.0 / (expected_home_goals * expected_away_goals)), 1])
    )
    LB = jnp.max(
        jnp.array(
            [jnp.max(-1.0 / expected_home_goals), jnp.max(-1.0 / expected_away_goals)]
        )
    )
    return LB, UB

def dixon_coles_correlation_term(
    home_goals: Union[int, Iterable[int]],
    away_goals: Union[int, Iterable[int]],
    home_rate: jnp.array,
    away_rate: jnp.array,
    corr_coef: jnp.array,
) -> jnp.array:
    # correlation term from dixon and coles paper
    if isinstance(home_goals, int):
        home_goals = np.array(home_goals).reshape((1,))
    if isinstance(away_goals, int):
        away_goals = np.array(away_goals).reshape((1,))

    corr_term = jnp.zeros_like(home_rate)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    corr_term = corr_term.at[..., nil_nil].set(
        jnp.log(
            1.0
            - corr_coef[..., None] * home_rate[..., nil_nil] * away_rate[..., nil_nil],
        )
    )

    one_nil = (home_goals == 1) & (away_goals == 0)
    corr_term = corr_term.at[..., one_nil].set(
        jnp.log(1.0 + corr_coef[..., None] * away_rate[..., one_nil]),
    )

    nil_one = (home_goals == 0) & (away_goals == 1)
    corr_term = corr_term.at[..., nil_one].set(
        jnp.log(1.0 + corr_coef[..., None] * home_rate[..., nil_one]),
    )

    one_one = (home_goals == 1) & (away_goals == 1)
    corr_term = corr_term.at[..., one_one].set(
        jnp.log(1.0 - corr_coef[..., None]),
    )

    return corr_term

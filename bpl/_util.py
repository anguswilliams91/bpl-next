"""Private utility functions."""
from typing import Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


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


def dixon_coles_correlation_term(
    home_goals: Union[int, Iterable[int]],
    away_goals: Union[int, Iterable[int]],
    home_rate: jnp.array,
    away_rate: jnp.array,
    corr_coef: jnp.array,  # rho in dixon and coles
    weights: Optional[jnp.array] = None,
    tol: float = 0,  # FIXME workaround to clip negative values to tol to avoid NaNs
) -> jnp.array:
    """
    Correlation term (tau) from Dixon and Coles paper.
    """
    if isinstance(home_goals, int):
        home_goals = np.array(home_goals).reshape((1,))
    if isinstance(away_goals, int):
        away_goals = np.array(away_goals).reshape((1,))

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
        ),
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

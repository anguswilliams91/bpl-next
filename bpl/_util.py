"""Private utility functions."""
from typing import Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np


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
            - corr_coef[..., None]
            * home_rate[..., nil_nil]
            * away_rate[..., nil_nil],
        )
    )

    one_nil = (home_goals == 1) & (away_goals == 0)
    corr_term = corr_term.at[..., one_nil].set(
        jnp.log(
            1.0 + corr_coef[..., None] * away_rate[..., one_nil]
        ),
    )

    nil_one = (home_goals == 0) & (away_goals == 1)
    corr_term = corr_term.at[..., nil_one].set(
        jnp.log(
            1.0 + corr_coef[..., None] * home_rate[..., nil_one]
        ),
    )

    one_one = (home_goals == 1) & (away_goals == 1)
    corr_term = corr_term.at[..., one_one].set(
        jnp.log(1.0 - corr_coef[..., None]),
    )

    return corr_term

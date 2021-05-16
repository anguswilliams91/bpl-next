"""Private utility functions."""
import jax
import jax.numpy as jnp
import numpy as np


def dixon_coles_correlation_term(
    home_goals: np.array,
    away_goals: np.array,
    home_rate: jnp.array,
    away_rate: jnp.array,
    corr_coef: jnp.array,
) -> jnp.array:
    # correlation term from dixon and coles paper
    corr_term = jnp.zeros_like(home_rate)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    corr_term = jax.ops.index_update(
        corr_term,
        (..., nil_nil),
        jnp.log(
            1.0
            - corr_coef[..., None] * home_rate[..., nil_nil] * away_rate[..., nil_nil]
        ),
    )

    one_nil = (home_goals == 1) & (away_goals == 0)
    corr_term = jax.ops.index_update(
        corr_term,
        (..., one_nil),
        jnp.log(1.0 + corr_coef[..., None] * away_rate[..., one_nil]),
    )

    nil_one = (home_goals == 0) & (away_goals == 1)
    corr_term = jax.ops.index_update(
        corr_term,
        (..., nil_one),
        jnp.log(1.0 + corr_coef[..., None] * home_rate[..., nil_one]),
    )

    one_one = (home_goals == 1) & (away_goals == 1)
    corr_term = jax.ops.index_update(
        corr_term, (..., one_one), jnp.log(1.0 - corr_coef[..., None])
    )

    return corr_term

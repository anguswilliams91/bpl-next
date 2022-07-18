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
    tol: float = 0,  # FIXME workaround to clip negative values to tol to avoid NaNs
) -> jnp.array:
    # correlation term from dixon and coles paper
    corr_term = jnp.zeros_like(home_rate)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    corr_term = corr_term.at[..., nil_nil].set(
        jnp.log(
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
        jnp.log(
            jnp.clip(1.0 + corr_coef[..., None] * away_rate[..., one_nil], a_min=tol)
        ),
    )

    nil_one = (home_goals == 0) & (away_goals == 1)
    corr_term = corr_term.at[..., nil_one].set(
        jnp.log(
            jnp.clip(1.0 + corr_coef[..., None] * home_rate[..., nil_one], a_min=tol)
        ),
    )

    one_one = (home_goals == 1) & (away_goals == 1)
    corr_term = corr_term.at[..., one_one].set(
        jnp.log(jnp.clip(1.0 - corr_coef[..., None], a_min=tol)),
    )

    return corr_term
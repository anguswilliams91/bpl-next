import jax.numpy as jnp
import pytest

from bpl_next.dixon_coles import DixonColesMatchPredictor


def test_fit(dummy_data):
    model = DixonColesMatchPredictor().fit(dummy_data)

    assert model.attack is not None
    assert model.defence is not None
    assert model.home_advantage is not None
    assert model.teams is not None
    assert model.corr_coef is not None

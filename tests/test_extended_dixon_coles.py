from bpl_next.extended_dixon_coles import ExtendedDixonColesMatchPredictor


def test_time_weighted_vs_not(timed_dummy_data):
    """
    Test that the fitted model parameters respond as expected to including or excluding
    time weighting.
    """
    # with no time weighting each team in the dummy data wins the same number of matches
    # so their attack and defence strengths should be the same
    model_no_time = ExtendedDixonColesMatchPredictor().fit(timed_dummy_data)
    attack_no_time = model_no_time.attack.mean(axis=0)
    assert abs(attack_no_time[1] - attack_no_time[0]) < 0.05
    defence_no_time = model_no_time.defence.mean(axis=0)
    assert abs(defence_no_time[1] - defence_no_time[0]) < 0.05

    # with time weighting team A winning the first matches and team B winning the last
    # matches in the dummy data matters, so team B should be rated stronger
    model_with_time = ExtendedDixonColesMatchPredictor().fit(
        timed_dummy_data, epsilon=1
    )
    attack_with_time = model_with_time.attack.mean(axis=0)
    assert (attack_with_time[1] - attack_with_time[0]) > 0.75
    defence_with_time = model_with_time.defence.mean(axis=0)
    assert abs(defence_with_time[1] - defence_with_time[0]) > 0.75


def test_epsilon(timed_dummy_data):
    """
    Test that the fitted model parameters respond as expected to vaying the epsilon
    value.
    """
    model_epsilon1 = ExtendedDixonColesMatchPredictor().fit(timed_dummy_data, epsilon=1)
    attack_epsilon1 = model_epsilon1.attack.mean(axis=0)
    delta_attack_1 = abs(attack_epsilon1[1] - attack_epsilon1[0])
    defence_epsilon1 = model_epsilon1.attack.mean(axis=0)
    delta_defence_1 = abs(defence_epsilon1[1] - defence_epsilon1[0])

    model_epsilon2 = ExtendedDixonColesMatchPredictor().fit(timed_dummy_data, epsilon=2)
    attack_epsilon2 = model_epsilon2.attack.mean(axis=0)
    delta_attack_2 = abs(attack_epsilon2[1] - attack_epsilon2[0])
    defence_epsilon2 = model_epsilon2.attack.mean(axis=0)
    delta_defence_2 = abs(defence_epsilon2[1] - defence_epsilon2[0])

    # increasing epsilon should increase the impact of time weighting
    assert delta_attack_2 > 1.5 * delta_attack_1
    assert delta_defence_2 > 1.5 * delta_defence_1

"""Basic sanity checks for the PeakSeekingEnv."""

from peak_seeking.envs.peak_seeking_env import PeakSeekingEnv


def test_reset_returns_valid_observation():
    env = PeakSeekingEnv(size=10)
    obs, _ = env.reset(seed=123)
    env.close()
    assert obs.shape == (2,)
    assert obs.dtype.kind == "i"
    assert obs.min() >= 0
    assert obs.max() < 10


def test_step_progresses_environment():
    env = PeakSeekingEnv(size=10)
    obs, _ = env.reset(seed=0)
    next_obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    env.close()
    assert next_obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

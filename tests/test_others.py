from envs.custom_envs import make_point2d_dads_env
from envs.skill_wrapper import SkillWrapper
import numpy as np

NUM_SKILLS = 3


def skill(obs):
    is_batch = obs.ndim > 1
    if is_batch:
        return obs[:, -NUM_SKILLS:]
    return obs[-NUM_SKILLS:]


def test_reset_does_not_change_skill():
    env = make_point2d_dads_env()
    wenv = SkillWrapper(env=env, num_latent_skills=NUM_SKILLS, skill_type="cont_uniform",
                        min_steps_before_resample=3,  resample_prob=1)
    obs = wenv.reset()
    many_obs = np.asarray([wenv.step(wenv.action_space.sample())[0] for _ in range(4)])
    assert np.allclose(skill(obs), skill(many_obs[:(3-1)]))
    assert not np.allclose(skill(obs), skill(many_obs[-1]))

    new_obs = wenv.reset()
    assert np.allclose(skill(many_obs[-1]), skill(new_obs))

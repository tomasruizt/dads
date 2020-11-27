import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from pytorch_mppi.mppi import MPPI
from tf_agents.trajectories.time_step import TimeStep

from common_funcs import SkillProvider
from envs.custom_envs import DADSEnv
from skill_dynamics import SkillDynamics


class MPPISkillProvider(SkillProvider):
    def __init__(self, env: DADSEnv, dynamics: SkillDynamics, skills_to_plan: int):
        self._env = env
        self._dynamics = dynamics
        action_dim = dynamics._action_size
        self._device = "cpu"
        self._planner = MPPI(dynamics=self._dynamics_fn,
                             running_cost=self._cost_fn,
                             nx=env.dyn_obs_dim(),
                             u_min=-torch.ones(action_dim), u_max=torch.ones(action_dim),
                             noise_sigma=0.1*torch.eye(action_dim), device=self._device, horizon=skills_to_plan, lambda_=1e-6)

    def _dynamics_fn(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        res = self._dynamics.predict_state(timesteps=state.cpu().numpy(), actions=actions.cpu().numpy())
        return torch.tensor(res, device=self._device)

    def _cost_fn(self, dyn_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cur_dyn_obs = dyn_obs.cpu().numpy()
        next_dyn_obs = self._dynamics.predict_state(timesteps=cur_dyn_obs, actions=actions.cpu().numpy())
        reward = self._env.compute_reward(cur_dyn_obs, next_dyn_obs, info=DADSEnv.OBS_TYPE.DYNAMICS_OBS)
        return torch.tensor(-reward, device=self._device)

    def start_episode(self):
        self._planner.reset()

    def get_skill(self, ts: TimeStep) -> np.ndarray:
        dyn_obs = self._env.to_dynamics_obs(ts.observation)
        return self._planner.command(state=dyn_obs).cpu().numpy()

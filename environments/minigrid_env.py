"""
environments/minigrid_env.py

MiniGrid environment wrapper implementing BaseEnvironment.

Wraps the Gymnasium-compatible MiniGrid environment and adds:
  - Ground-truth state extraction at every step (agent pos, direction, goal pos, carrying)
  - Flat grid tokenisation (no VQ-VAE needed — observations are already discrete)
  - Deterministic episode seeding for reproducibility

MiniGrid observation format (fully observable):
  - env.unwrapped.grid: full Grid object with all objects
  - env.unwrapped.agent_pos: (x, y) tuple
  - env.unwrapped.agent_dir: int in {0,1,2,3}

Tokenisation:
  The full grid is encoded as a (H, W, 3) uint8 array by MiniGrid,
  where channels are [object_type, colour, state]. We flatten to (H*W*3,)
  and treat each integer as a discrete token. This directly follows
  the Othello-GPT protocol (Li et al., 2023) — sequence of integers, no embeddings needed
  beyond a standard lookup table in the transformer.
"""

from typing import Any

import numpy as np

from environments.base_env import BaseEnvironment, Timestep


class MiniGridEnv(BaseEnvironment):
    """
    Wrapper around MiniGrid Gymnasium environments.

    Supported env_id values (from minigrid package):
        MiniGrid-FourRooms-v0      (default — richer navigation)
        MiniGrid-Empty-8x8-v0      (sanity check — simplest possible)
        MiniGrid-DoorKey-8x8-v0    (key-lock interaction)
        MiniGrid-MultiRoom-N4-S5-v0

    Ground-truth state logged per step:
        agent_x, agent_y:  agent grid coordinates
        agent_direction:   facing direction (0=right,1=down,2=left,3=up)
        goal_x, goal_y:    goal object grid coordinates (-1 if no goal in env)
        carrying:          1 if agent is carrying an object, 0 otherwise
    """

    # MiniGrid encodes each cell as (object_type, colour, state) — 3 channels
    # Object type IDs: 0=unseen,1=empty,2=wall,3=floor,4=door,5=key,6=ball,7=box,8=goal,9=lava,10=agent
    CHANNELS_PER_CELL = 3

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed)
        self._env = None
        self._current_obs = None
        self._goal_pos = (-1, -1)   # cached at reset; goal doesn't move mid-episode
        self._build_env()

    def _validate_config(self) -> None:
        required = ["name", "max_steps", "fully_observable"]
        missing = [k for k in required if k not in self.config.get("environment", {})]
        if missing:
            raise ValueError(f"MiniGridEnv config missing keys: {missing}")

    def _build_env(self) -> None:
        """Construct the underlying Gymnasium environment."""
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
        except ImportError as e:
            raise ImportError(
                "minigrid and gymnasium are required. "
                "Run: pip install minigrid gymnasium"
            ) from e

        env_cfg = self.config["environment"]
        env_id = env_cfg["name"]
        max_steps = env_cfg["max_steps"]

        self._env = gym.make(
            env_id,
            max_steps=max_steps,
            render_mode=None,   # no rendering during data collection
        )

        # Wrap with FullyObsWrapper to get complete grid observations
        if env_cfg.get("fully_observable", True):
            from minigrid.wrappers import FullyObsWrapper
            self._env = FullyObsWrapper(self._env)

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Timestep:
        use_seed = seed if seed is not None else self.seed
        obs_dict, info = self._env.reset(seed=use_seed)

        # FullyObsWrapper returns a dict with key 'image': (H, W, 3) uint8
        obs_array = obs_dict["image"]
        self._current_obs = self._flatten_observation(obs_array)
        self._goal_pos = self._find_goal_position()

        return Timestep(
            observation=self._current_obs.copy(),
            state=self.get_state(),
            action=0,
            reward=0.0,
            done=False,
            info=info,
        )

    def step(self, action: int) -> Timestep:
        obs_dict, reward, terminated, truncated, info = self._env.step(action)

        obs_array = obs_dict["image"]
        self._current_obs = self._flatten_observation(obs_array)

        return Timestep(
            observation=self._current_obs.copy(),
            state=self.get_state(),
            action=action,
            reward=float(reward),
            done=terminated or truncated,
            info=info,
        )

    def get_state(self) -> dict[str, Any]:
        """
        Extract ground-truth state from the unwrapped MiniGrid environment.

        These values are used as probe targets — they must reflect the
        true underlying simulator state, not the observation encoding.
        """
        unwrapped = self._env.unwrapped

        agent_x, agent_y = unwrapped.agent_pos
        agent_dir = int(unwrapped.agent_dir)
        goal_x, goal_y = self._goal_pos
        carrying = 1 if unwrapped.carrying is not None else 0

        return {
            "agent_x": int(agent_x),
            "agent_y": int(agent_y),
            "agent_direction": agent_dir,
            "goal_x": int(goal_x),
            "goal_y": int(goal_y),
            "carrying": carrying,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple:
        """
        Flattened grid observation shape: (H * W * 3,).
        Computed from the actual environment grid dimensions.
        """
        h = self._env.unwrapped.height
        w = self._env.unwrapped.width
        return (h * w * self.CHANNELS_PER_CELL,)

    @property
    def action_space_size(self) -> int:
        return int(self._env.action_space.n)

    @property
    def state_variable_names(self) -> list[str]:
        return ["agent_x", "agent_y", "agent_direction", "goal_x", "goal_y", "carrying"]

    @property
    def env_name(self) -> str:
        return self.config["environment"]["name"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flatten_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Flatten (H, W, 3) uint8 grid to a 1D integer token sequence.

        Each cell contributes 3 integer tokens (object_type, colour, state).
        Values are already small integers — no additional encoding needed.
        The transformer's embedding layer handles the rest.
        """
        return obs.flatten().astype(np.int64)

    def _find_goal_position(self) -> tuple[int, int]:
        """
        Scan the grid for the goal object and return its (x, y) coordinates.
        Returns (-1, -1) if no goal exists in this environment.

        Called once at reset — goal position is static within an episode.
        """
        from minigrid.core.constants import OBJECT_TO_IDX
        goal_type_id = OBJECT_TO_IDX.get("goal", -1)

        grid = self._env.unwrapped.grid
        width = self._env.unwrapped.width
        height = self._env.unwrapped.height

        for x in range(width):
            for y in range(height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return (x, y)

        return (-1, -1)

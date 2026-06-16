"""
environments/base_env.py

Abstract base class for all dissertation environments.
Both MiniGrid and PyBox2D environments implement this interface,
ensuring the data pipeline and probe suite work identically for both.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Timestep:
    """
    One step of environment interaction.

    observation: raw model input (tokenised or pixel, depending on env).
                 Shape varies by environment — documented in subclass.
    state:       ground-truth state dict used as probe targets.
                 Keys are the state variable names from config YAML.
    action:      action taken at this step (integer).
    reward:      scalar reward (may be 0 for most steps in goal-seeking envs).
    done:        whether this is the final step of the episode.
    info:        optional extra metadata from the environment.
    """
    observation: np.ndarray
    state: dict[str, Any]
    action: int
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    A complete episode: list of Timesteps plus episode-level metadata.

    timesteps:    ordered list of all Timesteps in the episode.
    env_name:     string identifier for the environment.
    episode_seed: seed used to initialise this episode (for reproducibility).
    total_reward: sum of rewards across all timesteps.
    """
    timesteps: list[Timestep]
    env_name: str
    episode_seed: int
    total_reward: float = 0.0

    def __len__(self) -> int:
        return len(self.timesteps)

    @property
    def observations(self) -> np.ndarray:
        """Stack all observations into a single array. Shape: (T, *obs_shape)."""
        return np.stack([t.observation for t in self.timesteps])

    @property
    def states(self) -> dict[str, np.ndarray]:
        """
        Collect all ground-truth states into per-variable arrays.
        Returns: dict mapping variable name -> array of shape (T,) or (T, D).
        """
        keys = self.timesteps[0].state.keys()
        return {
            k: np.array([t.state[k] for t in self.timesteps])
            for k in keys
        }

    @property
    def actions(self) -> np.ndarray:
        """Shape: (T,)."""
        return np.array([t.action for t in self.timesteps])


class BaseEnvironment(ABC):
    """
    Abstract interface for dissertation environments.

    Subclasses must implement:
        reset()       -> Timestep   (returns initial observation + state)
        step(action)  -> Timestep   (advances environment by one step)
        get_state()   -> dict       (returns ground-truth state at current step)
        close()       -> None       (cleanup)

    And must define:
        observation_shape: tuple    (shape of a single observation array)
        action_space_size: int      (number of discrete actions)
        state_variable_names: list  (keys returned by get_state())
        env_name: str               (human-readable identifier)
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Subclasses should override this to check for required config keys.
        Called once at construction time — fail fast rather than deep in training.
        """
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> Timestep:
        """
        Reset the environment to an initial state.

        Args:
            seed: if provided, overrides the instance seed for this episode.

        Returns:
            Timestep with the initial observation and ground-truth state.
            action=0, reward=0.0, done=False by convention at reset.
        """

    @abstractmethod
    def step(self, action: int) -> Timestep:
        """
        Take one action in the environment.

        Args:
            action: integer index into the action space.

        Returns:
            Timestep with observation, ground-truth state, reward, and done flag.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Return the ground-truth state at the current timestep.

        This is the core contract: whatever the model sees as observations,
        this method gives us the true underlying state for probe targets.
        Keys must match state_variable_names.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the environment."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        """Shape of a single observation array (not batched)."""

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of discrete actions available."""

    @property
    @abstractmethod
    def state_variable_names(self) -> list[str]:
        """
        Names of all ground-truth state variables returned by get_state().
        Must be consistent across all calls for a given environment.
        """

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Human-readable environment identifier (e.g. 'minigrid', 'physics')."""

    def collect_trajectory(
        self,
        policy: "callable",
        max_steps: int,
        episode_seed: int | None = None,
    ) -> Trajectory:
        """
        Collect a single trajectory using the given policy.

        Args:
            policy:       callable(observation) -> action (int)
            max_steps:    maximum number of steps before forced termination
            episode_seed: seed for this specific episode

        Returns:
            Trajectory containing all Timesteps.
        """
        seed = episode_seed if episode_seed is not None else self.seed
        first_step = self.reset(seed=seed)
        timesteps = [first_step]
        total_reward = first_step.reward

        for _ in range(max_steps - 1):
            action = policy(first_step.observation)
            step = self.step(action)
            timesteps.append(step)
            total_reward += step.reward
            if step.done:
                break
            first_step = step

        return Trajectory(
            timesteps=timesteps,
            env_name=self.env_name,
            episode_seed=seed,
            total_reward=total_reward,
        )

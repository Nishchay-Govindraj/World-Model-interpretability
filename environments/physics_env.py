"""
environments/physics_env.py

Custom 2D Newtonian physics sandbox built on Pymunk (Chipmunk backend).
Implements BaseEnvironment.

Pymunk replaces PyBox2D — identical physics capabilities for our purposes:
  - Rigid body dynamics (gravity, velocity, angular velocity)
  - Elastic collisions with restitution
  - Friction
  - Contact detection via collision handlers

Design:
  - Rectangular walled arena with 4 static boundary segments
  - N rigid body circles (configurable, default 3)
  - Each episode: objects spawned at random non-overlapping positions
  - Actions: 8 directional impulses x N objects + 1 no-op

Observation:
  Rendered as (64, 64, 3) uint8 RGB image per step -> VQ-VAE tokeniser downstream.
  Ground-truth state is logged separately and never seen by the model.

Ground-truth state per step per object i:
  pos_x_{i}, pos_y_{i}     : position (pixels)
  vel_x_{i}, vel_y_{i}     : velocity (pixels/s)
  angle_{i}                : rotation angle (radians)
  angular_vel_{i}          : angular velocity (radians/s)
  in_contact_{i}           : binary contact flag
"""

from typing import Any

import numpy as np

from environments.base_env import BaseEnvironment, Timestep


class PhysicsEnv(BaseEnvironment):
    """
    Pymunk physics sandbox with N rigid body circles in a walled arena.

    Action space: 8 directions x N objects + 1 no-op
        Actions 0..7:   impulse on object 0 (N,NE,E,SE,S,SW,W,NW)
        Actions 8..15:  impulse on object 1
        ...
        Last action:    no-op

    Observation: (obs_size, obs_size, 3) uint8 rendered RGB frame (pre VQ-VAE)
    """

    DIRECTIONS = [
        ( 0.0,   1.0),    # N
        ( 0.707, 0.707),  # NE
        ( 1.0,   0.0),    # E
        ( 0.707,-0.707),  # SE
        ( 0.0,  -1.0),    # S
        (-0.707,-0.707),  # SW
        (-1.0,  0.0),     # W
        (-0.707, 0.707),  # NW
    ]
    NUM_DIRECTIONS = len(DIRECTIONS)
    IMPULSE_MAGNITUDE = 300.0   # pymunk uses pixel-scale units

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed)
        self._rng = np.random.default_rng(seed)
        self._cfg = config["environment"]
        self._num_objects = self._cfg["num_objects"]
        self._obs_size = config["tokenisation"]["obs_dim"]
        self._max_steps = config["data_collection"]["steps_per_trajectory"]
        self._dt = 1.0 / self._cfg["fps"]
        self._step_count = 0

        # World dimensions in pixels
        self._W = int(self._cfg["world_width"] * 30)   # 30px per metre
        self._H = int(self._cfg["world_height"] * 30)

        # Pymunk state
        self._space = None
        self._bodies = []
        self._shapes = []
        self._contacts: set[int] = set()   # indices of objects currently in contact

        self._build_space()

    def _validate_config(self) -> None:
        required = ["world_width", "world_height", "gravity", "fps",
                    "substeps", "num_objects"]
        env_cfg = self.config.get("environment", {})
        missing = [k for k in required if k not in env_cfg]
        if missing:
            raise ValueError(f"PhysicsEnv config missing keys: {missing}")

    def _build_space(self) -> None:
        """Initialise pymunk space with gravity."""
        try:
            import pymunk
        except ImportError as e:
            raise ImportError("pymunk is required. Run: pip install pymunk") from e

        import pymunk
        self._pymunk = pymunk

        self._space = pymunk.Space()
        self._space.gravity = (0, -self._cfg["gravity"] * 30)  # scale to pixels
        self._space.damping = 0.99   # slight damping to keep simulation stable

    def _add_walls(self) -> None:
        """Create 4 static boundary segments forming the arena."""
        W, H = self._W, self._H
        walls = [
            ((0, 0),   (W, 0)),    # bottom
            ((0, H),   (W, H)),    # top
            ((0, 0),   (0, H)),    # left
            ((W, 0),   (W, H)),    # right
        ]
        for a, b in walls:
            seg = self._pymunk.Segment(self._space.static_body, a, b, 2)
            seg.elasticity = self._cfg["object"]["restitution"]
            seg.friction = self._cfg["object"]["friction"]
            self._space.add(seg)

    def _spawn_objects(self) -> None:
        """Spawn N circles at random non-overlapping positions."""
        obj_cfg = self._cfg["object"]
        W, H = self._W, self._H

        # Remove old bodies
        for body in self._bodies:
            for shape in body.shapes:
                self._space.remove(shape)
            self._space.remove(body)
        self._bodies = []
        self._shapes = []
        self._contacts = set()

        placed = []   # list of (x, y, radius)
        radius_px_range = (
            int(obj_cfg["radius_range"][0] * 30),
            int(obj_cfg["radius_range"][1] * 30),
        )

        for i in range(self._num_objects):
            radius = int(self._rng.uniform(*radius_px_range))
            mass = float(self._rng.uniform(*obj_cfg["mass_range"]))
            moment = self._pymunk.moment_for_circle(mass, 0, radius)

            # Sample non-overlapping position
            for _ in range(200):
                x = float(self._rng.uniform(radius + 10, W - radius - 10))
                y = float(self._rng.uniform(radius + 10, H - radius - 10))
                overlap = any(
                    np.hypot(x - px, y - py) < (radius + pr + 5)
                    for (px, py, pr) in placed
                )
                if not overlap:
                    break

            placed.append((x, y, radius))

            body = self._pymunk.Body(mass, moment)
            body.position = (x, y)
            body.data = i   # store object index for contact lookup

            shape = self._pymunk.Circle(body, radius)
            shape.elasticity = obj_cfg["restitution"]
            shape.friction = obj_cfg["friction"]
            shape.collision_type = i + 1   # unique collision type per object

            self._space.add(body, shape)
            self._bodies.append(body)
            self._shapes.append(shape)

        # Register collision handlers for all object pairs
        self._register_collision_handlers()

    def _register_collision_handlers(self) -> None:
        """Set up collision handlers to track which objects are in contact.

        Pymunk 7.x API: space.on_collision(type_a, type_b, begin=..., separate=...)
        Collision types: objects use 1..N, walls use collision_type=0 (default static).
        """
        for i in range(self._num_objects):
            # Object-object collisions
            for j in range(i + 1, self._num_objects):
                def make_obj_handlers(_i: int, _j: int):
                    def begin(arbiter, space, data):
                        self._contacts.add(_i)
                        self._contacts.add(_j)
                        return True
                    def separate(arbiter, space, data):
                        self._contacts.discard(_i)
                        self._contacts.discard(_j)
                    return begin, separate

                begin_cb, separate_cb = make_obj_handlers(i, j)
                self._space.on_collision(
                    i + 1, j + 1,
                    begin=begin_cb,
                    separate=separate_cb,
                )

            # Object-wall collisions (walls have default collision_type=0)
            def make_wall_handlers(_i: int):
                def begin(arbiter, space, data):
                    self._contacts.add(_i)
                    return True
                def separate(arbiter, space, data):
                    self._contacts.discard(_i)
                return begin, separate

            wall_begin_cb, wall_separate_cb = make_wall_handlers(i)
            self._space.on_collision(
                i + 1, 0,
                begin=wall_begin_cb,
                separate=wall_separate_cb,
            )

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Timestep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0

        # Rebuild space cleanly
        self._space = self._pymunk.Space()
        self._space.gravity = (0, -self._cfg["gravity"] * 30)
        self._space.damping = 0.99
        self._bodies = []
        self._shapes = []
        self._contacts = set()

        self._add_walls()
        self._spawn_objects()

        # Settle
        for _ in range(10):
            self._space.step(self._dt / 10)

        return Timestep(
            observation=self._render(),
            state=self.get_state(),
            action=0,
            reward=0.0,
            done=False,
        )

    def step(self, action: int) -> Timestep:
        no_op = self._num_objects * self.NUM_DIRECTIONS
        if action < no_op:
            obj_idx = action // self.NUM_DIRECTIONS
            dir_idx = action % self.NUM_DIRECTIONS
            self._apply_impulse(obj_idx, dir_idx)

        substeps = self._cfg["substeps"]
        for _ in range(substeps):
            self._space.step(self._dt / substeps)

        self._step_count += 1
        done = self._step_count >= self._max_steps

        return Timestep(
            observation=self._render(),
            state=self.get_state(),
            action=action,
            reward=0.0,
            done=done,
        )

    def get_state(self) -> dict[str, Any]:
        state = {}
        for i, body in enumerate(self._bodies):
            state[f"pos_x_{i}"]       = float(body.position.x)
            state[f"pos_y_{i}"]       = float(body.position.y)
            state[f"vel_x_{i}"]       = float(body.velocity.x)
            state[f"vel_y_{i}"]       = float(body.velocity.y)
            state[f"angle_{i}"]        = float(body.angle)
            state[f"angular_vel_{i}"] = float(body.angular_velocity)
            state[f"in_contact_{i}"]  = int(i in self._contacts)
        return state

    def close(self) -> None:
        self._bodies = []
        self._shapes = []
        self._space = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple:
        return (self._obs_size, self._obs_size, 3)

    @property
    def action_space_size(self) -> int:
        return self._num_objects * self.NUM_DIRECTIONS + 1

    @property
    def state_variable_names(self) -> list[str]:
        names = []
        for i in range(self._num_objects):
            names.extend([
                f"pos_x_{i}", f"pos_y_{i}",
                f"vel_x_{i}", f"vel_y_{i}",
                f"angle_{i}", f"angular_vel_{i}",
                f"in_contact_{i}",
            ])
        return names

    @property
    def env_name(self) -> str:
        return "PhysicsSandbox-v0"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_impulse(self, obj_idx: int, dir_idx: int) -> None:
        if obj_idx >= len(self._bodies):
            return
        body = self._bodies[obj_idx]
        dx, dy = self.DIRECTIONS[dir_idx]
        body.apply_impulse_at_local_point(
            (dx * self.IMPULSE_MAGNITUDE, dy * self.IMPULSE_MAGNITUDE),
            (0, 0),
        )

    def _render(self) -> np.ndarray:
        """
        Render current state to (obs_size, obs_size, 3) uint8 RGB array.

        Fast numpy renderer — no matplotlib overhead.
        Draws directly into a pixel array using a vectorised filled-circle
        algorithm. ~100x faster than the matplotlib approach.

        Rendering:
          - Black background
          - White 1px border for arena walls
          - Each object drawn as a solid filled circle in a fixed colour
          - Physics coordinates scaled to obs_size
          - Y-axis flipped: pymunk y increases upward, image y increases downward
        """
        size = self._obs_size
        W, H = self._W, self._H

        scale_x = size / W
        scale_y = size / H

        # Fixed colours per object index (RGB uint8)
        COLOURS = [
            (79,  195, 247),   # blue   - object 0
            (255, 112,  67),   # orange - object 1
            (102, 187, 106),   # green  - object 2
        ]

        # Black canvas
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        # White border for arena walls
        canvas[0, :]  = 255
        canvas[-1, :] = 255
        canvas[:, 0]  = 255
        canvas[:, -1] = 255

        # Precompute pixel coordinate grids
        ys, xs = np.mgrid[0:size, 0:size]

        for i, (body, shape) in enumerate(zip(self._bodies, self._shapes)):
            # Convert pymunk coords to image coords (flip y axis)
            px = body.position.x * scale_x
            py = size - (body.position.y * scale_y)
            r  = shape.radius * scale_x

            # Vectorised filled circle
            mask = (xs - px) ** 2 + (ys - py) ** 2 <= r ** 2
            canvas[mask] = COLOURS[i % len(COLOURS)]

        return canvas

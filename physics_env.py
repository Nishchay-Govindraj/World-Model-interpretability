"""
environments/physics_env.py

Custom 2D Newtonian physics sandbox built on PyBox2D.
Implements BaseEnvironment.

Design goals:
  - Rich causal interactions: gravity, elastic collisions, friction
  - Clean ground-truth state: per-object position, velocity, angle, contact
  - Discrete action space: apply impulses to specific objects in specific directions
  - Deterministic seeding for reproducibility

Environment layout:
  - Rectangular walled arena (4 static boundary walls)
  - N rigid body circles (configurable, default 3)
  - Each episode: objects spawned at random non-overlapping positions
  - Actions: 8 directional impulses x N objects = 8*N discrete actions
             + 1 no-op = 8*N + 1 total actions

Observation:
  Rendered as a (64, 64, 3) uint8 RGB image per step,
  then passed to a VQ-VAE tokeniser to produce discrete tokens.
  Ground-truth state is logged separately and never seen by the model.

Contact detection:
  PyBox2D's contact listener fires callbacks when bodies begin/end contact.
  We track active contacts per body per step for the 'in_contact' state variable.
"""

from typing import Any

import numpy as np

from environments.base_env import BaseEnvironment, Timestep


class ContactListener:
    """
    Tracks which bodies are currently in contact.
    Registered with the Box2D world to receive collision callbacks.
    """

    def __init__(self):
        self.active_contacts: set[int] = set()  # set of body user_data IDs

    def BeginContact(self, contact) -> None:
        body_a = contact.fixtureA.body.userData
        body_b = contact.fixtureB.body.userData
        if body_a is not None:
            self.active_contacts.add(body_a)
        if body_b is not None:
            self.active_contacts.add(body_b)

    def EndContact(self, contact) -> None:
        body_a = contact.fixtureA.body.userData
        body_b = contact.fixtureB.body.userData
        # Only remove if no other contacts involve this body
        # Box2D handles multiple simultaneous contacts; we clear conservatively
        if body_a is not None:
            self.active_contacts.discard(body_a)
        if body_b is not None:
            self.active_contacts.discard(body_b)

    def PreSolve(self, contact, old_manifold) -> None:
        pass

    def PostSolve(self, contact, impulse) -> None:
        pass

    def reset(self) -> None:
        self.active_contacts.clear()


class PhysicsEnv(BaseEnvironment):
    """
    PyBox2D physics sandbox with N rigid body circles in a walled arena.

    Action space: 8 directions x N objects + 1 no-op
        Actions 0..7:   impulse on object 0 (N,NE,E,SE,S,SW,W,NW)
        Actions 8..15:  impulse on object 1
        ...
        Last action:    no-op

    Observation: (64, 64, 3) uint8 rendered RGB frame (pre-VQ-VAE)

    Ground-truth state per step (for N objects, indexed 0..N-1):
        pos_x_{i}, pos_y_{i}:       position
        vel_x_{i}, vel_y_{i}:       velocity
        angle_{i}:                  rotation angle (radians)
        angular_vel_{i}:            angular velocity
        in_contact_{i}:             binary contact flag
    """

    # 8 cardinal + intercardinal directions for impulse actions
    DIRECTIONS = [
        (0.0,   1.0),   # N
        (0.707, 0.707), # NE
        (1.0,   0.0),   # E
        (0.707,-0.707), # SE
        (0.0,  -1.0),   # S
        (-0.707,-0.707),# SW
        (-1.0,  0.0),   # W
        (-0.707, 0.707),# NW
    ]
    NUM_DIRECTIONS = len(DIRECTIONS)
    IMPULSE_MAGNITUDE = 5.0         # Newton-seconds

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed)
        self._rng = np.random.default_rng(seed)
        self._world = None
        self._bodies = []
        self._walls = []
        self._contact_listener = ContactListener()
        self._step_count = 0
        self._max_steps = config["data_collection"]["steps_per_trajectory"]
        self._cfg = config["environment"]
        self._num_objects = self._cfg["num_objects"]
        self._obs_size = config["tokenisation"]["obs_dim"]

        self._build_world()

    def _validate_config(self) -> None:
        required_env = ["world_width", "world_height", "gravity", "fps",
                        "substeps", "num_objects"]
        env_cfg = self.config.get("environment", {})
        missing = [k for k in required_env if k not in env_cfg]
        if missing:
            raise ValueError(f"PhysicsEnv config missing keys under 'environment': {missing}")

    def _build_world(self) -> None:
        """Initialise the Box2D world with gravity and contact listener."""
        try:
            import Box2D
            from Box2D.b2 import world as b2World, vec2
        except ImportError as e:
            raise ImportError(
                "Box2D is required. Run: pip install Box2D"
            ) from e

        from Box2D.b2 import world as b2World, vec2

        self._b2_world = b2World(
            gravity=vec2(0, self._cfg["gravity"]),
            doSleep=True,
        )
        self._b2_world.contactListener = self._contact_listener

    def _add_walls(self) -> None:
        """Create 4 static wall bodies forming the arena boundary."""
        from Box2D.b2 import staticBody, polygonShape

        w = self._cfg["world_width"]
        h = self._cfg["world_height"]
        thickness = 0.5

        wall_specs = [
            # (centre_x, centre_y, half_width, half_height)
            (w / 2, -thickness / 2, w / 2, thickness / 2),    # bottom
            (w / 2, h + thickness / 2, w / 2, thickness / 2), # top
            (-thickness / 2, h / 2, thickness / 2, h / 2),    # left
            (w + thickness / 2, h / 2, thickness / 2, h / 2), # right
        ]

        self._walls = []
        for cx, cy, hx, hy in wall_specs:
            body = self._b2_world.CreateStaticBody(position=(cx, cy))
            body.CreatePolygonFixture(box=(hx, hy), friction=self._cfg["friction"])
            self._walls.append(body)

    def _spawn_objects(self) -> None:
        """
        Spawn N circular rigid bodies at random non-overlapping positions.
        Each body is assigned an integer ID (its index) via userData.
        """
        from Box2D.b2 import dynamicBody, circleShape

        obj_cfg = self._cfg["object"]
        w = self._cfg["world_width"]
        h = self._cfg["world_height"]

        # Remove any existing dynamic bodies
        for body in self._bodies:
            self._b2_world.DestroyBody(body)
        self._bodies = []

        placed_positions = []

        for i in range(self._num_objects):
            radius = self._rng.uniform(*obj_cfg["radius_range"])
            mass = self._rng.uniform(*obj_cfg["mass_range"])

            # Sample non-overlapping position (max 100 attempts per object)
            for _ in range(100):
                x = self._rng.uniform(radius + 1, w - radius - 1)
                y = self._rng.uniform(radius + 1, h - radius - 1)
                overlap = any(
                    np.hypot(x - px, y - py) < (radius + pr + 0.2)
                    for (px, py, pr) in placed_positions
                )
                if not overlap:
                    break

            placed_positions.append((x, y, radius))

            body = self._b2_world.CreateDynamicBody(position=(x, y))
            fixture = body.CreateCircleFixture(
                radius=radius,
                density=mass / (np.pi * radius ** 2),
                friction=obj_cfg["friction"],
                restitution=obj_cfg["restitution"],
            )
            body.userData = i  # used by ContactListener
            self._bodies.append(body)

    def reset(self, seed: int | None = None) -> Timestep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            use_seed = seed
        else:
            use_seed = self.seed

        self._contact_listener.reset()
        self._step_count = 0

        # Rebuild walls and objects for clean state
        for wall in self._walls:
            self._b2_world.DestroyBody(wall)
        self._walls = []
        self._add_walls()
        self._spawn_objects()

        # Advance one substep to settle initial contacts
        self._b2_world.Step(
            1.0 / self._cfg["fps"],
            velocityIterations=8,
            positionIterations=3,
        )

        obs = self._render()
        return Timestep(
            observation=obs,
            state=self.get_state(),
            action=0,
            reward=0.0,
            done=False,
        )

    def step(self, action: int) -> Timestep:
        # Apply impulse if action is not the no-op
        no_op_action = self._num_objects * self.NUM_DIRECTIONS
        if action < no_op_action:
            obj_idx = action // self.NUM_DIRECTIONS
            dir_idx = action % self.NUM_DIRECTIONS
            self._apply_impulse(obj_idx, dir_idx)

        # Step physics with substeps for numerical stability
        dt = 1.0 / self._cfg["fps"]
        for _ in range(self._cfg["substeps"]):
            self._b2_world.Step(
                dt / self._cfg["substeps"],
                velocityIterations=8,
                positionIterations=3,
            )

        self._step_count += 1
        done = self._step_count >= self._max_steps

        obs = self._render()
        return Timestep(
            observation=obs,
            state=self.get_state(),
            action=action,
            reward=0.0,  # no reward signal — prediction objective only
            done=done,
        )

    def get_state(self) -> dict[str, Any]:
        """
        Return ground-truth physical state for all objects.

        Variable naming: {variable_name}_{object_index}
        e.g. pos_x_0, vel_y_2, in_contact_1
        """
        state = {}
        for i, body in enumerate(self._bodies):
            state[f"pos_x_{i}"]       = float(body.position.x)
            state[f"pos_y_{i}"]       = float(body.position.y)
            state[f"vel_x_{i}"]       = float(body.linearVelocity.x)
            state[f"vel_y_{i}"]       = float(body.linearVelocity.y)
            state[f"angle_{i}"]        = float(body.angle)
            state[f"angular_vel_{i}"] = float(body.angularVelocity)
            state[f"in_contact_{i}"]  = int(i in self._contact_listener.active_contacts)
        return state

    def close(self) -> None:
        # Box2D doesn't require explicit cleanup, but clear references
        self._bodies = []
        self._walls = []
        self._b2_world = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple:
        """(H, W, C) — rendered RGB frame before VQ-VAE tokenisation."""
        size = self._obs_size
        return (size, size, 3)

    @property
    def action_space_size(self) -> int:
        """8 directions x N objects + 1 no-op."""
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
        """Apply an instantaneous impulse to the specified object."""
        if obj_idx >= len(self._bodies):
            return
        body = self._bodies[obj_idx]
        dx, dy = self.DIRECTIONS[dir_idx]
        impulse = (dx * self.IMPULSE_MAGNITUDE, dy * self.IMPULSE_MAGNITUDE)
        body.ApplyLinearImpulse(impulse, body.worldCenter, wake=True)

    def _render(self) -> np.ndarray:
        """
        Render the current world state to a (H, W, 3) uint8 RGB array.

        Uses matplotlib for simplicity. In production, switch to a headless
        Cairo/Pillow renderer for speed if data collection becomes a bottleneck.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        size = self._obs_size
        w = self._cfg["world_width"]
        h = self._cfg["world_height"]

        fig, ax = plt.subplots(figsize=(size / 50, size / 50), dpi=50)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        colours = ["#4FC3F7", "#FF7043", "#66BB6A"]  # one colour per object
        for i, body in enumerate(self._bodies):
            # Get radius from first circle fixture
            radius = body.fixtures[0].shape.radius
            circle = plt.Circle(
                (body.position.x, body.position.y),
                radius,
                color=colours[i % len(colours)],
                zorder=2,
            )
            ax.add_patch(circle)

        # Draw walls as grey rectangles
        rect = patches.Rectangle(
            (0, 0), w, h,
            linewidth=2, edgecolor="grey", facecolor="none", zorder=3,
        )
        ax.add_patch(rect)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )
        plt.close(fig)

        # Drop alpha channel and resize to target obs_size
        img_rgb = img[:, :, :3]
        if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
            from PIL import Image
            img_rgb = np.array(
                Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR)
            )

        return img_rgb.astype(np.uint8)

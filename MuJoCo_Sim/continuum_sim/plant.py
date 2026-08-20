"""
MuJoCo plant wrapper - the independent robot that the controller talks to.

This is the object that replaces the PCC forward-kinematics "simulation" in
continuum_ellipse.py. The controller sends a tendon-displacement command u and
gets back a measured tip pose; nothing about how that pose was produced comes
from the controller's own model.

    plant = ContinuumPlant()
    plant.reset()
    pose = plant.step(u)          # u = [u1, u2, u3] metres -> [x, y, psi] SI

Deliberately mirrors the interface the GUI already expects, so the same wrapper
can also feed the existing UDP pose path (pose_feedback.py) unchanged.

Author: Badhon Kumar
"""

import mujoco
import numpy as np

from . import build_model
from . import params as P


class ContinuumPlant:
    """Quasi-static MuJoCo plant for the planar tendon-driven continuum arm."""

    def __init__(self,
                 gravity=None,
                 stiffness=None,
                 seg_stiffness_scale=None,
                 tendon_kp=None,
                 links_per_seg=None,
                 settle_time=2.0,
                 pos_tol=1e-7,
                 stable_steps=25,
                 noise_pos=0.0,
                 noise_psi=0.0,
                 seed=0):
        """
        noise_pos / noise_psi add Gaussian measurement noise (m / rad) to the
        returned pose, to mimic the real camera or EM tracker. Off by default.
        """
        self.xml = build_model.build_xml(
            gravity=gravity,
            stiffness=stiffness,
            seg_stiffness_scale=seg_stiffness_scale,
            tendon_kp=tendon_kp,
            links_per_seg=links_per_seg,
        )
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.n_per_seg = links_per_seg or P.LINKS_PER_SEG

        # Equilibrium is judged on TIP MOTION, not joint velocity.
        #
        # max|qvel| is not a usable criterion for this model: the stiff tendon
        # position actuators sustain a bounded numerical buzz of ~0.25 rad/s that
        # never decays, while the tip itself is stable to ~3 um. Testing qvel
        # therefore reports "never settled" for an arm that has completely
        # stopped, and led to closed-loop runs where only 14% of steps were
        # counted as settled. Tip displacement is both the observable the
        # controller actually consumes and the one that genuinely converges.
        self.settle_time = settle_time
        self.pos_tol = pos_tol
        self.stable_steps = stable_steps
        self.noise_pos = noise_pos
        self.noise_psi = noise_psi
        self.rng = np.random.default_rng(seed)

        self._tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self._cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "planar")

        mujoco.mj_forward(self.model, self.data)
        # Rest lengths are read from the compiled model, not assumed: the
        # routing radius steps at each segment boundary add ~1.5 mm per crossing.
        self.rest = P.rest_lengths_from_model(self.model, self.data, mujoco)
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.rest
        mujoco.mj_forward(self.model, self.data)
        self.diverged = False
        return self.tip_pose()

    def tip_pose(self, noisy: bool = False):
        """True tip pose [x, y, psi] in SI. Set noisy=True to emulate a sensor."""
        R = self.data.site_xmat[self._tip].reshape(3, 3)
        p = np.array([self.data.site_xpos[self._tip][0],
                      self.data.site_xpos[self._tip][1],
                      np.arctan2(R[1, 0], R[0, 0])])
        if noisy and (self.noise_pos > 0 or self.noise_psi > 0):
            p = p + self.rng.normal(0.0, [self.noise_pos, self.noise_pos,
                                          self.noise_psi])
        return p

    def segment_angles(self):
        """Total bend angle of each segment (rad), for comparison with PCC."""
        n = self.n_per_seg
        return np.array([float(np.sum(self.data.qpos[s * n:(s + 1) * n]))
                         for s in range(P.N_SEG)])

    def tendon_lengths(self):
        return np.asarray(self.data.ten_length).copy()

    # ── actuation ────────────────────────────────────────────────────────────
    def step(self, u, settle_time=None, noisy=False):
        """
        Command u = [u1, u2, u3] (metres of tendon displacement) and integrate
        to quasi-static equilibrium, then return the measured tip pose.

        The paper assumes quasi-static motion, so the plant is always read at
        equilibrium rather than mid-transient. If it fails to settle within the
        budget the last state is returned and `settled` is False.
        """
        self.data.ctrl[:] = P.u_to_tendon_lengths(np.asarray(u, float), self.rest)
        t = self.settle_time if settle_time is None else settle_time
        n = int(t / self.model.opt.timestep)
        self.settled = False
        prev = self.tip_pose()[:2]
        stable = 0
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)):
                self.diverged = True
                break
            cur = self.tip_pose()[:2]
            stable = stable + 1 if np.hypot(*(cur - prev)) < self.pos_tol else 0
            prev = cur
            if stable >= self.stable_steps:
                self.settled = True
                self.settle_steps = i + 1
                break
        else:
            self.settle_steps = n
        return self.tip_pose(noisy=noisy)

    # ── rendering ────────────────────────────────────────────────────────────
    def render(self, width=900, height=600, renderer=None):
        own = renderer is None
        r = renderer or mujoco.Renderer(self.model, height=height, width=width)
        try:
            r.update_scene(self.data, camera=self._cam)
            return r.render()
        finally:
            if own:
                r.close()

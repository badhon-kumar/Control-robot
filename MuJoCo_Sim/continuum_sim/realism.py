"""
Optional physical feedback-loop effects for the MuJoCo continuum plant.

The default is deliberately ideal/off. The "physical" preset is not a random
curve generator: it is a compact actuator/sensor model with parameters that
can be identified from hardware logs later:

  * motor speed limit and first-order drive dynamics,
  * tendon backlash/friction/compliance expressed as lost tendon motion,
  * fourth-order low-pass filtering on the measured pose,
  * small residual tracker noise and sample delay.

Without hardware logs this is a physics-based calibration, not an experimental
identification. The parameter names and units are chosen so replacing the
defaults with measured values is straightforward.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from pose_feedback import ButterworthLowPass4
except Exception:  # pragma: no cover - import path differs in standalone use
    ButterworthLowPass4 = None


class _CascadedLowPass4:
    """Dependency-free four-pole low-pass fallback."""

    def __init__(self, cutoff_hz: float, sample_hz: float, channels: int):
        self.channels = int(channels)
        dt = 1.0 / float(sample_hz)
        tau = 1.0 / (2.0 * np.pi * float(cutoff_hz))
        self._alpha = dt / (tau + dt)
        self._state = np.zeros((4, self.channels), dtype=float)
        self._ready = False

    def reset(self, value):
        v = np.asarray(value, float).reshape(self.channels)
        self._state[:, :] = v
        self._ready = True

    def update(self, value):
        v = np.asarray(value, float).reshape(self.channels)
        if not self._ready:
            self.reset(v)
            return v.copy()
        cur = v
        for i in range(4):
            self._state[i] += self._alpha * (cur - self._state[i])
            cur = self._state[i]
        return cur.copy()


@dataclass(frozen=True)
class RealismConfig:
    enabled: bool = False
    sample_hz: float = 20.0            # controller/measurement sample rate

    # Sensor path, in the same units as the controller consumes.
    sensor_pos_noise: float = 0.0      # m RMS residual marker/camera noise
    sensor_psi_noise: float = 0.0      # rad RMS residual attitude noise
    sensor_delay_steps: int = 0        # control samples
    sensor_cutoff_hz: float = 0.0      # 4th-order low-pass cutoff, 0 disables

    # Backwards-compatible aliases used by older checks/scripts. These are not
    # used by the physical preset.
    sensor_pos_bias: float = 0.0
    sensor_psi_bias: float = 0.0
    sensor_bias_alpha: float = 0.98
    sensor_filter_alpha: float = 0.0

    # Actuator/tendon path. All values are expressed in tendon displacement.
    actuator_noise: float = 0.0        # m RMS residual unmodelled drive noise
    actuator_deadband: float = 0.0     # m; alias for tendon_backlash if set
    actuator_lag: float = 0.0          # 0 none, alias for drive lag if set
    actuator_hysteresis: float = 0.0   # m; alias for direction loss if set
    motor_speed_mps: float = 0.0       # m/s max tendon command slew, 0 disables
    drive_tau_s: float = 0.0           # first-order motor/drive response
    tendon_backlash_m: float = 0.0     # m lost before a changed pull takes up
    tendon_friction_m: float = 0.0     # m direction-dependent guide/capstan loss
    tendon_compliance: float = 0.0     # fraction of command lost to cable stretch
    segment_gain: tuple = (1.0, 1.0, 1.0)
    seed: int = 0

    @property
    def active(self) -> bool:
        return (
            self.enabled
            and (
                self.sensor_pos_noise > 0.0
                or self.sensor_psi_noise > 0.0
                or self.sensor_pos_bias > 0.0
                or self.sensor_psi_bias > 0.0
                or self.sensor_cutoff_hz > 0.0
                or self.sensor_filter_alpha > 0.0
                or self.sensor_delay_steps > 0
                or self.actuator_noise > 0.0
                or self.actuator_deadband > 0.0
                or self.actuator_lag > 0.0
                or self.actuator_hysteresis > 0.0
                or self.motor_speed_mps > 0.0
                or self.drive_tau_s > 0.0
                or self.tendon_backlash_m > 0.0
                or self.tendon_friction_m > 0.0
                or self.tendon_compliance > 0.0
                or tuple(self.segment_gain) != (1.0, 1.0, 1.0)
            )
        )


PRESETS = {
    "off": RealismConfig(),
    "light": RealismConfig(
        enabled=True,
        sensor_pos_noise=0.00008,
        sensor_psi_noise=np.deg2rad(0.05),
        sensor_cutoff_hz=2.0,
        motor_speed_mps=0.040,
        drive_tau_s=0.04,
        tendon_backlash_m=0.00003,
        tendon_friction_m=0.00002,
        tendon_compliance=0.015,
    ),
    "physical": RealismConfig(
        enabled=True,
        sample_hz=20.0,
        sensor_pos_noise=0.00012,
        sensor_psi_noise=np.deg2rad(0.08),
        sensor_delay_steps=0,
        sensor_cutoff_hz=0.0,
        motor_speed_mps=0.040,
        drive_tau_s=0.12,
        tendon_backlash_m=0.00025,
        tendon_friction_m=0.00020,
        tendon_compliance=0.070,
    ),
}
PRESETS["paper"] = PRESETS["physical"]


def preset(name: str) -> RealismConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown realism preset {name!r}; choose from {', '.join(PRESETS)}")


def describe(config: RealismConfig) -> str:
    if not config.active:
        return "off"
    filter_desc = (
        f"4th-order low-pass {config.sensor_cutoff_hz:.1f} Hz at "
        f"{config.sample_hz:.0f} Hz"
        if config.sensor_cutoff_hz > 0.0
        else "sensor low-pass off"
    )
    return (
        f"sensor residual {config.sensor_pos_noise * 1000:.2f} mm / "
        f"{np.rad2deg(config.sensor_psi_noise):.2f} deg, "
        f"{filter_desc}, "
        f"delay {config.sensor_delay_steps} step(s), "
        f"motor speed {config.motor_speed_mps * 1000:.1f} mm/s, "
        f"drive tau {config.drive_tau_s * 1000:.0f} ms, "
        f"backlash {config.tendon_backlash_m * 1000:.2f} mm, "
        f"friction loss {config.tendon_friction_m * 1000:.2f} mm, "
        f"compliance {config.tendon_compliance * 100:.1f}%"
    )


class RealismLayer:
    """Stateful sensor/actuator imperfection model used inside feedback loops."""

    def __init__(self, config: RealismConfig | None = None):
        self.config = config or RealismConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._pose_buffer = deque()
        self._last_applied_u = None
        self._sensor_bias = np.zeros(3)
        self._filtered_pose = None
        self._last_direction = np.zeros(3)
        self._last_requested_u = None
        self._pose_filter = None

    @property
    def active(self) -> bool:
        return self.config.active

    def reset(self, initial_pose=None, initial_u=None, initial_applied_u=None):
        self._pose_buffer.clear()
        self._sensor_bias[:] = 0.0
        self._filtered_pose = None
        self._last_direction[:] = 0.0
        if initial_u is not None:
            gain = np.asarray(self.config.segment_gain, float)
            u = np.asarray(initial_u, float).copy()
            self._last_requested_u = u * gain if gain.shape == u.shape else u
        else:
            self._last_requested_u = None
        self._pose_filter = None
        if initial_pose is not None:
            p = np.asarray(initial_pose, float).copy()
            for _ in range(max(0, self.config.sensor_delay_steps) + 1):
                self._pose_buffer.append(p.copy())
            if self.config.sensor_cutoff_hz > 0.0 and ButterworthLowPass4 is not None:
                filter_cls = ButterworthLowPass4 or _CascadedLowPass4
                self._pose_filter = filter_cls(
                    cutoff_hz=self.config.sensor_cutoff_hz,
                    sample_hz=self.config.sample_hz,
                    channels=3,
                )
                self._pose_filter.reset(p)
            elif self.config.sensor_cutoff_hz > 0.0:
                self._pose_filter = _CascadedLowPass4(
                    cutoff_hz=self.config.sensor_cutoff_hz,
                    sample_hz=self.config.sample_hz,
                    channels=3,
                )
                self._pose_filter.reset(p)
        if initial_applied_u is not None:
            self._last_applied_u = np.asarray(initial_applied_u, float).copy()
        else:
            self._last_applied_u = (
                None if initial_u is None else np.asarray(initial_u, float).copy()
            )

    def command_to_plant(self, requested_u):
        """Convert controller-requested u into the imperfect plant command."""
        requested = np.asarray(requested_u, float).copy()
        u = requested.copy()
        if not self.active:
            self._last_applied_u = u.copy()
            self._last_requested_u = u.copy()
            return u

        gain = np.asarray(self.config.segment_gain, float)
        if gain.shape == u.shape:
            u = u * gain
        target_u = u.copy()

        if self._last_requested_u is not None and self.config.motor_speed_mps > 0.0:
            max_du = self.config.motor_speed_mps / max(self.config.sample_hz, 1e-9)
            u = self._last_requested_u + np.clip(u - self._last_requested_u,
                                                 -max_du, max_du)
            target_u = u.copy()

        deadband = max(self.config.actuator_deadband, self.config.tendon_backlash_m)
        if self._last_applied_u is not None and deadband > 0.0:
            du = u - self._last_applied_u
            hold = np.abs(du) < deadband
            u[hold] = self._last_applied_u[hold]

        if self._last_applied_u is not None and (
            self.config.drive_tau_s > 0.0 or self.config.actuator_lag > 0.0
        ):
            if self.config.drive_tau_s > 0.0:
                dt = 1.0 / max(self.config.sample_hz, 1e-9)
                a = np.exp(-dt / self.config.drive_tau_s)
            else:
                a = self.config.actuator_lag
            a = float(np.clip(a, 0.0, 0.98))
            u = a * self._last_applied_u + (1.0 - a) * u

        direction_loss = max(self.config.actuator_hysteresis, self.config.tendon_friction_m)
        if self._last_applied_u is not None and direction_loss > 0.0:
            du = u - self._last_applied_u
            moving = np.abs(du) >= max(deadband, 1e-12)
            direction = self._last_direction.copy()
            direction[moving] = np.sign(du[moving])
            u = u - direction * direction_loss
            self._last_direction = direction

        if self.config.tendon_compliance > 0.0:
            u = u * (1.0 - float(np.clip(self.config.tendon_compliance, 0.0, 0.5)))

        if self.config.actuator_noise > 0.0:
            u += self.rng.normal(0.0, self.config.actuator_noise, size=u.shape)

        self._last_applied_u = u.copy()
        self._last_requested_u = target_u.copy()
        return u

    def measure(self, true_pose):
        """Return the delayed/noisy pose seen by the controller."""
        p = np.asarray(true_pose, float).copy()
        if not self.active:
            return p

        if self.config.sensor_pos_bias > 0.0 or self.config.sensor_psi_bias > 0.0:
            alpha = float(np.clip(self.config.sensor_bias_alpha, 0.0, 0.999))
            sigma = np.array([
                self.config.sensor_pos_bias,
                self.config.sensor_pos_bias,
                self.config.sensor_psi_bias,
            ])
            innovation = sigma * np.sqrt(max(0.0, 1.0 - alpha * alpha))
            self._sensor_bias = (
                alpha * self._sensor_bias
                + self.rng.normal(0.0, innovation)
            )
            p += self._sensor_bias

        if self.config.sensor_pos_noise > 0.0 or self.config.sensor_psi_noise > 0.0:
            p += self.rng.normal(
                0.0,
                [self.config.sensor_pos_noise,
                 self.config.sensor_pos_noise,
                 self.config.sensor_psi_noise],
                )

        if self.config.sensor_cutoff_hz > 0.0 and self._pose_filter is not None:
            p = self._pose_filter.update(p)

        if self.config.sensor_filter_alpha > 0.0:
            alpha = float(np.clip(self.config.sensor_filter_alpha, 0.0, 0.98))
            if self._filtered_pose is None:
                self._filtered_pose = p.copy()
            else:
                self._filtered_pose = alpha * self._filtered_pose + (1.0 - alpha) * p
            p = self._filtered_pose.copy()

        if not self._pose_buffer:
            self.reset(initial_pose=p)
        self._pose_buffer.append(p.copy())
        while len(self._pose_buffer) > max(0, self.config.sensor_delay_steps) + 1:
            self._pose_buffer.popleft()
        return self._pose_buffer[0].copy()

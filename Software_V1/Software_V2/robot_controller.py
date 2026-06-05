"""
robot_controller.py
===================
6-Axis Robot Controller for MyActuator RMD-X8-120 Motors

This module manages all six motors as a coordinated robotic arm system.
It handles:
  - Synchronized multi-axis motion
  - Real-time feedback monitoring
  - Safety limits and emergency stop
  - Pre-defined motion sequences (test routines)

Joint Mapping (customize to match your robot's physical configuration):
  Joint 1 (Base)     → Motor ID 1
  Joint 2 (Shoulder) → Motor ID 2
  Joint 3 (Elbow)    → Motor ID 3
  Joint 4 (Wrist 1)  → Motor ID 4
  Joint 5 (Wrist 2)  → Motor ID 5
  Joint 6 (Tool)     → Motor ID 6
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional

from can_interface import CANBus
from motor_driver import RMDMotor, MotorFeedback

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Joint Configuration
# ──────────────────────────────────────────────

@dataclass
class JointConfig:
    """Per-joint limits and parameters."""
    motor_id: int
    name: str
    min_deg: float      # Software minimum position limit (degrees)
    max_deg: float      # Software maximum position limit (degrees)
    home_deg: float     # Home / zero position (degrees)
    max_speed_dps: float = 180.0   # Default max speed for this joint


# Default joint configuration for a 6-DOF arm
# Adjust these limits to match your physical robot!
DEFAULT_JOINT_CONFIG = [
    JointConfig(motor_id=1, name="Base",     min_deg=-180.0, max_deg=180.0,  home_deg=0.0,  max_speed_dps=120.0),
    JointConfig(motor_id=2, name="Shoulder", min_deg=-90.0,  max_deg=90.0,   home_deg=0.0,  max_speed_dps=90.0),
    JointConfig(motor_id=3, name="Elbow",    min_deg=-120.0, max_deg=120.0,  home_deg=0.0,  max_speed_dps=90.0),
    JointConfig(motor_id=4, name="Wrist1",   min_deg=-180.0, max_deg=180.0,  home_deg=0.0,  max_speed_dps=180.0),
    JointConfig(motor_id=5, name="Wrist2",   min_deg=-90.0,  max_deg=90.0,   home_deg=0.0,  max_speed_dps=180.0),
    JointConfig(motor_id=6, name="Tool",     min_deg=-360.0, max_deg=360.0,  home_deg=0.0,  max_speed_dps=200.0),
]


# ──────────────────────────────────────────────
# Robot Controller
# ──────────────────────────────────────────────

class RobotController:
    """
    High-level 6-axis robot controller.

    Parameters
    ----------
    bus : CANBus
        Shared CAN bus instance.
    joint_configs : list[JointConfig]
        Per-joint configuration. Defaults to DEFAULT_JOINT_CONFIG.
    monitor_hz : float
        Frequency of the background feedback monitoring loop (Hz).
    """

    def __init__(
        self,
        bus: CANBus,
        joint_configs: list[JointConfig] = None,
        monitor_hz: float = 50.0,
    ):
        self.bus = bus
        self.joint_configs = joint_configs or DEFAULT_JOINT_CONFIG
        self.monitor_hz = monitor_hz
        self._estop = False

        # Create motor driver instances
        self.motors: dict[int, RMDMotor] = {}
        for cfg in self.joint_configs:
            self.motors[cfg.motor_id] = RMDMotor(
                motor_id=cfg.motor_id,
                bus=bus,
                max_current_a=25.0,
            )

        # Config dict for easy lookup
        self._cfg: dict[int, JointConfig] = {c.motor_id: c for c in self.joint_configs}

        # Background monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        self._feedback: dict[int, MotorFeedback] = {}
        self._feedback_lock = threading.Lock()

        logger.info(f"RobotController: Initialized with {len(self.motors)} joints.")

    # ── Lifecycle ─────────────────────────────

    def start(self):
        """Enable all motors and start feedback monitoring."""
        logger.info("RobotController: Starting...")
        for motor_id, motor in self.motors.items():
            motor.run()
            logger.info(f"  Motor {motor_id} ({self._cfg[motor_id].name}): enabled")

        self._start_monitor()
        logger.info("RobotController: Ready.")

    def stop_all(self):
        """Send stop command to every motor immediately."""
        logger.info("RobotController: Stopping all motors.")
        for motor in self.motors.values():
            motor.stop()

    def estop(self):
        """
        EMERGENCY STOP: Immediately cut torque on all motors.
        Must call reset() to re-enable after an e-stop.
        """
        self._estop = True
        logger.critical("EMERGENCY STOP ACTIVATED")
        for motor in self.motors.values():
            motor.turn_off()

    def reset(self):
        """Clear emergency stop and re-enable motors."""
        self._estop = False
        for motor in self.motors.values():
            motor.run()
        logger.info("RobotController: Emergency stop cleared. Motors re-enabled.")

    def close(self):
        """Shutdown: stop all motors and monitoring thread."""
        self._stop_monitor()
        self.stop_all()
        self.bus.close()
        logger.info("RobotController: Shutdown complete.")

    # ── Safety check ─────────────────────────

    def _check_limit(self, motor_id: int, position_deg: float) -> bool:
        """Returns True if position is within configured joint limits."""
        cfg = self._cfg.get(motor_id)
        if cfg is None:
            return True
        if not cfg.min_deg <= position_deg <= cfg.max_deg:
            logger.error(
                f"Joint {motor_id} ({cfg.name}): Position {position_deg:.1f}° "
                f"exceeds limits [{cfg.min_deg}°, {cfg.max_deg}°]"
            )
            return False
        return True

    # ── Single joint commands ─────────────────

    def move_joint(
        self,
        motor_id: int,
        position_deg: float,
        speed_dps: float = None,
        wait: bool = False,
        timeout_s: float = 15.0,
    ) -> bool:
        """
        Move a single joint to an absolute position.

        Parameters
        ----------
        motor_id : int
            Joint motor ID (1–6).
        position_deg : float
            Target position in degrees.
        speed_dps : float
            Speed in degrees/second. Uses joint default if None.
        wait : bool
            Block until motion completes.
        """
        if self._estop:
            logger.error("Cannot move: Emergency stop is active. Call reset() first.")
            return False
        if not self._check_limit(motor_id, position_deg):
            return False

        motor = self.motors.get(motor_id)
        if motor is None:
            logger.error(f"Motor {motor_id} not found.")
            return False

        cfg = self._cfg[motor_id]
        speed = speed_dps or cfg.max_speed_dps

        return motor.set_position(position_deg, speed, wait=wait, timeout_s=timeout_s)

    # ── Synchronized multi-joint motion ───────

    def move_all_joints(
        self,
        positions: dict[int, float],
        speed_dps: float = None,
        wait: bool = True,
        tolerance_deg: float = 1.5,
        timeout_s: float = 20.0,
    ) -> bool:
        """
        Move multiple joints simultaneously to target positions.

        Parameters
        ----------
        positions : dict[int, float]
            {motor_id: target_position_deg} for each joint to move.
        speed_dps : float
            Common speed for all joints. Uses per-joint defaults if None.
        wait : bool
            Block until all joints reach targets.
        tolerance_deg : float
            Position error for arrival detection (degrees).
        timeout_s : float
            Maximum wait time (seconds).

        Returns
        -------
        bool : True if all joints reached targets (when wait=True).
        """
        if self._estop:
            logger.error("Cannot move: Emergency stop is active.")
            return False

        # Validate all positions before sending any commands
        for motor_id, pos in positions.items():
            if not self._check_limit(motor_id, pos):
                return False

        # Send commands to all joints (non-blocking, near-simultaneous)
        for motor_id, target_pos in positions.items():
            motor = self.motors.get(motor_id)
            cfg = self._cfg.get(motor_id)
            if motor and cfg:
                speed = speed_dps or cfg.max_speed_dps
                motor.set_position(target_pos, max_speed_dps=speed, wait=False)
                time.sleep(0.001)   # 1 ms inter-command gap (CAN bus scheduling)

        if not wait:
            return True

        # Wait for all joints to arrive
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            all_arrived = True
            for motor_id, target_pos in positions.items():
                motor = self.motors.get(motor_id)
                if motor is None:
                    continue
                current = motor.get_position()
                if abs(current - target_pos) > tolerance_deg:
                    all_arrived = False
                    break
            if all_arrived:
                logger.info("move_all_joints: All joints arrived at targets.")
                return True
            time.sleep(0.05)

        logger.warning("move_all_joints: Timeout waiting for joints to arrive.")
        return False

    def go_home(self, speed_dps: float = 90.0, wait: bool = True) -> bool:
        """Move all joints to their configured home positions."""
        logger.info("RobotController: Moving to HOME position.")
        home_positions = {cfg.motor_id: cfg.home_deg for cfg in self.joint_configs}
        return self.move_all_joints(home_positions, speed_dps=speed_dps, wait=wait)

    # ── Feedback monitoring ───────────────────

    def _start_monitor(self):
        """Start the background feedback polling loop."""
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="FeedbackMonitor"
        )
        self._monitor_thread.start()

    def _stop_monitor(self):
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background loop: polls each motor for feedback at monitor_hz."""
        interval = 1.0 / self.monitor_hz
        motor_ids = list(self.motors.keys())
        idx = 0

        while self._monitor_running:
            motor_id = motor_ids[idx % len(motor_ids)]
            motor = self.motors[motor_id]

            fb = motor.read_status()
            if fb:
                with self._feedback_lock:
                    self._feedback[motor_id] = fb

                # Temperature safety check
                if fb.temperature_c > 75.0:
                    logger.warning(
                        f"Motor {motor_id}: HIGH TEMPERATURE {fb.temperature_c:.1f}°C"
                    )
                if fb.temperature_c > 85.0:
                    logger.critical(
                        f"Motor {motor_id}: CRITICAL TEMPERATURE — triggering E-STOP"
                    )
                    self.estop()

            idx += 1
            time.sleep(interval / len(motor_ids))

    def get_all_feedback(self) -> dict[int, MotorFeedback]:
        """Return the latest feedback from all motors."""
        with self._feedback_lock:
            return dict(self._feedback)

    def print_status(self):
        """Print a formatted status table of all joints."""
        fb_all = self.get_all_feedback()
        print("\n" + "─" * 75)
        print(f"{'Joint':<10} {'Name':<12} {'Position':>10} {'Velocity':>10} {'Current':>8} {'Temp':>7}")
        print("─" * 75)
        for cfg in self.joint_configs:
            fb = fb_all.get(cfg.motor_id)
            if fb:
                print(
                    f"  J{cfg.motor_id:<7} {cfg.name:<12} "
                    f"{fb.position_deg:>8.2f}°  "
                    f"{fb.velocity_dps:>7.1f}°/s  "
                    f"{fb.current_a:>5.2f}A  "
                    f"{fb.temperature_c:>5.1f}°C"
                )
            else:
                print(f"  J{cfg.motor_id:<7} {cfg.name:<12}  [no data]")
        print("─" * 75 + "\n")

    # ── Context manager ───────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()

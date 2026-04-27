"""
motor_driver.py
===============
MyActuator RMD-X8-120 Motor Driver (CAN Protocol Implementation)

This module implements the complete CAN command set for the MyActuator
RMD-X8-120 motors as documented in the MyActuator CAN Protocol v2.x manual.

Each motor has a unique CAN ID: 0x141 + (motor_id - 1)
  Motor 1 → ID 0x141
  Motor 2 → ID 0x142
  ...
  Motor 6 → ID 0x146

All commands are 8 bytes. All responses are also 8 bytes.

Reference: MyActuator RMD-X Series CAN Protocol Documentation
"""

import struct
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from can_interface import CANBus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class MotorFeedback:
    """Parsed feedback data from a motor response frame."""
    motor_id: int
    temperature_c: float = 0.0       # Motor winding temperature (°C)
    current_a: float = 0.0           # Phase current (Amps)
    velocity_dps: float = 0.0        # Angular velocity (degrees/second)
    position_deg: float = 0.0        # Multi-turn position (degrees)
    position_raw: int = 0            # Raw encoder value (for debugging)
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        return (
            f"Motor {self.motor_id}: "
            f"Pos={self.position_deg:8.2f}°  "
            f"Vel={self.velocity_dps:7.1f}°/s  "
            f"Cur={self.current_a:5.2f}A  "
            f"Temp={self.temperature_c:4.1f}°C"
        )


@dataclass
class MotorStatus:
    """Current known status of a motor (updated from feedback)."""
    motor_id: int
    is_enabled: bool = False
    last_feedback: Optional[MotorFeedback] = None
    error_count: int = 0


# ──────────────────────────────────────────────
# Command Byte Constants (from MyActuator protocol)
# ──────────────────────────────────────────────

class CMD:
    # Motor state control
    MOTOR_OFF           = 0x80   # Turn motor off (stops drive, no position loss)
    MOTOR_STOP          = 0x81   # Stop motor (clears running state)
    MOTOR_RUN           = 0x88   # Resume from stop

    # Read commands
    READ_PID            = 0x30   # Read PID parameters
    READ_ENCODER        = 0x90   # Read encoder position
    READ_MULTI_TURNS    = 0x92   # Read multi-turn position, speed, current
    READ_STATUS_1       = 0x9A   # Read temperature, voltage, error flags
    READ_STATUS_2       = 0x9C   # Read temperature, current, speed, position

    # Write commands
    WRITE_ENCODER_ZERO  = 0x91   # Set current position as zero point
    WRITE_PID_RAM       = 0x31   # Write PID params to RAM (temporary)
    WRITE_PID_ROM       = 0x32   # Write PID params to ROM (permanent)

    # Position control
    POS_CTRL_MODE1      = 0xA3   # Absolute position, default speed (6 bytes pos)
    POS_CTRL_MODE2      = 0xA4   # Absolute position, configurable speed
    POS_CTRL_INCR       = 0xA7   # Incremental (relative) position control
    POS_CTRL_MODE3      = 0x9D   # Single-turn position control

    # Velocity control
    VEL_CTRL            = 0xA2   # Velocity / speed control

    # Torque / current control
    TORQUE_CTRL         = 0xA1   # Open-loop torque / current control


# ──────────────────────────────────────────────
# Motor Driver Class
# ──────────────────────────────────────────────

class RMDMotor:
    """
    Driver for a single MyActuator RMD-X8-120 motor.

    Parameters
    ----------
    motor_id : int
        Motor ID set via MyActuator software (1–32).
        The CAN arbitration ID = 0x140 + motor_id
    bus : CANBus
        Shared CAN bus instance.
    max_current_a : float
        Software current limit (Amps). RMD-X8-120 peak is ~33A.
    """

    # Protocol constants
    CURRENT_SCALE = 2048.0 / 33.0   # raw per Amp  (33A → 2048 raw)
    SPEED_SCALE   = 1.0              # 1 raw = 1 dps (degrees per second)
    POS_SCALE     = 100.0            # raw per degree (0.01° resolution)

    def __init__(self, motor_id: int, bus: CANBus, max_current_a: float = 20.0):
        if not 1 <= motor_id <= 32:
            raise ValueError(f"Motor ID must be 1–32, got {motor_id}")
        self.motor_id = motor_id
        self.bus = bus
        self.max_current_a = max_current_a
        self.can_id = 0x140 + motor_id    # e.g., Motor 1 → 0x141
        self.status = MotorStatus(motor_id=motor_id)

    # ── Low-level frame builders ──────────────

    def _send(self, cmd: int, payload: list[int] = None) -> bool:
        """Build and send an 8-byte CAN frame."""
        data = [cmd] + (payload or [0x00] * 7)
        data = data[:8]                     # truncate to 8 bytes
        data += [0x00] * (8 - len(data))    # pad to 8 bytes
        logger.debug(f"Motor {self.motor_id} → CAN 0x{self.can_id:03X}: "
                     f"{' '.join(f'{b:02X}' for b in data)}")
        return self.bus.send(self.can_id, data)

    def _parse_feedback(self, msg) -> Optional[MotorFeedback]:
        """Parse an 8-byte motor response into a MotorFeedback object."""
        if msg is None:
            # In simulation mode, poll the simulated state instead
            sim = self.bus.get_simulated_state(self.motor_id)
            if sim:
                fb = MotorFeedback(
                    motor_id=self.motor_id,
                    temperature_c=sim.temperature_c,
                    current_a=sim.current_a,
                    velocity_dps=sim.velocity_dps,
                    position_deg=sim.position_deg,
                )
                self.status.last_feedback = fb
                return fb
            return None

        data = msg.data
        if len(data) < 8:
            return None

        cmd = data[0]
        if cmd in (CMD.READ_MULTI_TURNS, CMD.POS_CTRL_MODE2, CMD.POS_CTRL_MODE1,
                   CMD.POS_CTRL_INCR, CMD.VEL_CTRL, CMD.READ_STATUS_2):
            temp_c       = data[1]
            current_raw  = struct.unpack_from('<h', data, 2)[0]   # signed 16-bit
            speed_raw    = struct.unpack_from('<h', data, 4)[0]   # signed 16-bit
            pos_raw      = struct.unpack_from('<H', data, 6)[0]   # unsigned 16-bit

            fb = MotorFeedback(
                motor_id=self.motor_id,
                temperature_c=float(temp_c),
                current_a=current_raw / self.CURRENT_SCALE,
                velocity_dps=float(speed_raw),
                position_deg=pos_raw * 0.01,   # 0.01°/LSB
                position_raw=pos_raw,
            )
            self.status.last_feedback = fb
            return fb
        return None

    # ── Motor state commands ──────────────────

    def turn_off(self) -> bool:
        """
        Turn off motor drive output.
        Motor loses torque but retains position in firmware memory.
        """
        logger.info(f"Motor {self.motor_id}: TURN OFF")
        self.status.is_enabled = False
        return self._send(CMD.MOTOR_OFF)

    def stop(self) -> bool:
        """
        Stop motor and clear running state.
        Motor will not hold position after this command.
        """
        logger.info(f"Motor {self.motor_id}: STOP")
        self.status.is_enabled = False
        return self._send(CMD.MOTOR_STOP)

    def run(self) -> bool:
        """Resume motor from stopped state."""
        logger.info(f"Motor {self.motor_id}: RUN")
        self.status.is_enabled = True
        return self._send(CMD.MOTOR_RUN)

    # ── Read commands ─────────────────────────

    def read_status(self) -> Optional[MotorFeedback]:
        """
        Request and return current position, velocity, current, and temperature.

        Returns MotorFeedback object, or None if communication fails.
        """
        self._send(CMD.READ_MULTI_TURNS)
        time.sleep(0.002)   # 2 ms round-trip time budget
        msg = self.bus.receive(timeout=0.05)
        return self._parse_feedback(msg)

    def get_position(self) -> float:
        """
        Convenience: return current position in degrees.
        Returns 0.0 if communication fails.
        """
        if self.bus.simulated:
            state = self.bus.get_simulated_state(self.motor_id)
            return state.position_deg if state else 0.0
        fb = self.read_status()
        return fb.position_deg if fb else 0.0

    # ── Position control commands ─────────────

    def set_position(
        self,
        position_deg: float,
        max_speed_dps: float = 180.0,
        wait: bool = False,
        tolerance_deg: float = 1.0,
        timeout_s: float = 10.0,
    ) -> bool:
        """
        Move motor to absolute position (multi-turn).

        Parameters
        ----------
        position_deg : float
            Target position in degrees (multi-turn, e.g., 720° = 2 full rotations).
        max_speed_dps : float
            Maximum speed for this move (degrees/second). Max is ~720 dps.
        wait : bool
            If True, block until position is reached or timeout.
        tolerance_deg : float
            Position error considered "arrived" (degrees).
        timeout_s : float
            Maximum wait time in seconds.

        Returns
        -------
        bool : True if command sent (and arrived, if wait=True).
        """
        # Clamp speed to safe range
        speed_dps = max(1.0, min(max_speed_dps, 720.0))
        speed_raw = int(speed_dps)   # 1 dps per LSB

        # Position in 0.01° units, signed 32-bit
        pos_raw = int(position_deg * self.POS_SCALE)

        # Pack: [CMD, 0x00, 0x00, 0x00, speed_hi, speed_lo, pos_hi, pos_lo]
        # Actually RMD protocol for 0xA4:
        # Byte 0: 0xA4
        # Byte 1: null
        # Byte 2: null
        # Byte 3: null
        # Byte 4-5: speed (uint16, dps)
        # Byte 6-7: position (int16 of 0.01° ... but MyActuator uses int32 for full range)
        # NOTE: For large multi-turn positions use the 4-byte position variant

        # # Using 4-byte absolute position (signed int32 in units of 0.01°)
        # pos_bytes = struct.pack('<i', pos_raw)   # little-endian signed 32-bit
        # speed_bytes = struct.pack('<H', speed_raw)

        # # Frame layout: [0xA4, 0, 0, 0, spd_hi, spd_lo, pos_b2, pos_b3]
        # # (Only lower 2 bytes of position in standard frame — for ±327° range)
        # # For full multi-turn: use READ_MULTI_TURNS command set
        # payload = [
        #     0x00,                  # byte 1: null
        #     0x00,                  # byte 2: null
        #     0x00,                  # byte 3: null
        #     speed_bytes[0],        # byte 4: speed high
        #     speed_bytes[1],        # byte 5: speed low
        #     pos_bytes[2],          # byte 6: position high byte (of int32)
        #     pos_bytes[3],          # byte 7: position low byte
        # ]

        # Pack into Little-Endian (<):
        # H = unsigned 16-bit int (2 bytes for speed)
        # i = signed 32-bit int (4 bytes for position)
        packed_data = struct.pack('<Hi', speed_raw, pos_raw)

        # RMD 0xA4 Frame: [CMD, 0x00, Speed(2 bytes), Position(4 bytes)]
        payload = [
            0x00,               # Byte 1: Null
            packed_data[0],     # Byte 2: Speed LSB
            packed_data[1],     # Byte 3: Speed MSB
            packed_data[2],     # Byte 4: Position LSB
            packed_data[3],     # Byte 5: Position Byte 2
            packed_data[4],     # Byte 6: Position Byte 3
            packed_data[5],     # Byte 7: Position MSB
        ]

        logger.info(f"Motor {self.motor_id}: SET POSITION {position_deg:.2f}° @ {speed_dps:.0f}°/s")

        # Update simulation target directly for full multi-turn support
        if self.bus.simulated:
            state = self.bus.get_simulated_state(self.motor_id)
            if state:
                state.target_position = position_deg

        ok = self._send(CMD.POS_CTRL_MODE2, payload)
        self.status.is_enabled = True

        if wait and ok:
            return self._wait_for_position(position_deg, tolerance_deg, timeout_s)
        return ok

    def set_position_relative(
        self,
        delta_deg: float,
        max_speed_dps: float = 180.0,
        wait: bool = False,
        timeout_s: float = 10.0,
    ) -> bool:
        """
        Move motor by a relative offset from current position.

        Parameters
        ----------
        delta_deg : float
            Angular offset in degrees (positive = CCW, negative = CW).
        """
        current = self.get_position()
        return self.set_position(current + delta_deg, max_speed_dps, wait, timeout_s=timeout_s)

    def _wait_for_position(
        self, target_deg: float, tolerance: float, timeout: float
    ) -> bool:
        """Block until motor reaches target or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self.get_position()
            if abs(current - target_deg) <= tolerance:
                return True
            time.sleep(0.05)
        logger.warning(
            f"Motor {self.motor_id}: Position timeout. "
            f"Target={target_deg:.1f}°, Current={self.get_position():.1f}°"
        )
        return False

    # ── Velocity control command ──────────────

    def set_velocity(self, velocity_dps: float) -> bool:
        """
        Set motor to run at constant velocity.

        Parameters
        ----------
        velocity_dps : float
            Target velocity in degrees/second.
            Positive = CCW, Negative = CW. Range: ±720 dps typically.
        """
        speed_raw = int(velocity_dps * 100)   # 0.01 dps per LSB
        speed_bytes = struct.pack('<i', speed_raw)   # signed 32-bit

        payload = [
            0x00, 0x00, 0x00,     # bytes 1-3: null
            speed_bytes[0], speed_bytes[1],
            speed_bytes[2], speed_bytes[3],
        ]

        logger.info(f"Motor {self.motor_id}: SET VELOCITY {velocity_dps:.1f}°/s")
        if self.bus.simulated:
            state = self.bus.get_simulated_state(self.motor_id)
            if state:
                state.target_velocity = velocity_dps

        return self._send(CMD.VEL_CTRL, payload)

    # ── Torque control command ────────────────

    def set_torque(self, current_a: float) -> bool:
        """
        Set motor torque via direct current control.

        Parameters
        ----------
        current_a : float
            Target phase current in Amps.
            Positive = CCW torque, Negative = CW torque.
            Clamped to ±max_current_a.
        """
        current_a = max(-self.max_current_a, min(self.max_current_a, current_a))
        current_raw = int(current_a * self.CURRENT_SCALE)
        current_bytes = struct.pack('<h', current_raw)   # signed 16-bit

        payload = [
            0x00, 0x00, 0x00, 0x00, 0x00,
            current_bytes[0], current_bytes[1],
        ]

        logger.info(f"Motor {self.motor_id}: SET TORQUE {current_a:.2f}A")
        return self._send(CMD.TORQUE_CTRL, payload)

    # ── Calibration ───────────────────────────

    def set_zero_position(self) -> bool:
        """
        Set the current position as the new zero/home reference.
        WARNING: This writes to motor ROM. Use only during calibration.
        """
        logger.warning(f"Motor {self.motor_id}: SETTING ZERO POSITION (writes to ROM)")
        return self._send(CMD.WRITE_ENCODER_ZERO)

    # ── String representation ─────────────────

    def __repr__(self):
        fb = self.status.last_feedback
        if fb:
            return f"RMDMotor(id={self.motor_id}, pos={fb.position_deg:.1f}°, enabled={self.status.is_enabled})"
        return f"RMDMotor(id={self.motor_id}, enabled={self.status.is_enabled})"

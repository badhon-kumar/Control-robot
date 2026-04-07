"""
can_interface.py
================
CAN Bus Interface Abstraction Layer for MyActuator RMD-X8-120 Motors

This module provides a unified interface that works in two modes:
  - SIMULATED: No hardware needed. Fakes responses for development/testing.
  - REAL:      Uses the 'python-can' library with a USB-to-CAN adapter
               (e.g., CANable, PEAK PCAN-USB, Kvaser, etc.)

Usage:
    bus = CANBus(simulated=True)   # for testing without hardware
    bus = CANBus(simulated=False)  # for real hardware (requires USB-CAN adapter)
"""

import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Simulated CAN Message (mirrors python-can API)
# ──────────────────────────────────────────────

class SimulatedMessage:
    """Mimics a python-can Message object for simulation purposes."""

    def __init__(self, arbitration_id: int, data: list[int]):
        self.arbitration_id = arbitration_id
        self.data = bytearray(data)
        self.timestamp = time.time()
        self.is_extended_id = False

    def __repr__(self):
        hex_data = " ".join(f"{b:02X}" for b in self.data)
        return f"SimMsg [ID=0x{self.arbitration_id:03X}] [{hex_data}]"


# ──────────────────────────────────────────────
# Simulated Motor State
# ──────────────────────────────────────────────

class SimulatedMotorState:
    """Tracks the internal state of a single simulated motor."""

    def __init__(self, motor_id: int):
        self.motor_id = motor_id
        self.position_deg = 0.0      # current position in degrees
        self.velocity_dps = 0.0      # current velocity in degrees/sec
        self.current_a = 0.0         # current draw in Amps
        self.temperature_c = 35.0    # motor temperature in Celsius
        self.target_position = 0.0
        self.target_velocity = 0.0

    def step(self, dt: float):
        """Simulate motor physics for one time step."""
        # Simple first-order system: position moves toward target
        error = self.target_position - self.position_deg
        self.velocity_dps = error * 5.0   # proportional response
        self.velocity_dps = max(-500.0, min(500.0, self.velocity_dps))
        self.position_deg += self.velocity_dps * dt

        # Simulate current proportional to velocity
        self.current_a = abs(self.velocity_dps) / 200.0

        # Simulate temperature rising with load
        self.temperature_c += self.current_a * 0.01 * dt
        self.temperature_c = max(25.0, min(80.0, self.temperature_c))

    def build_feedback_frame(self) -> SimulatedMessage:
        """Build a fake feedback CAN frame matching RMD-X8 response format."""
        # RMD-X8 position feedback: command byte + temperature + current + speed + position
        temp_raw = int(self.temperature_c)
        current_raw = int(self.current_a * 100) & 0xFFFF
        speed_raw = int(self.velocity_dps) & 0xFFFF
        position_raw = int((self.position_deg / 360.0) * 16384) & 0xFFFF  # 14-bit encoder

        data = [
            0x92,                          # Position control feedback command
            temp_raw & 0xFF,               # Temperature
            (current_raw >> 8) & 0xFF,     # Current high byte
            current_raw & 0xFF,            # Current low byte
            (speed_raw >> 8) & 0xFF,       # Speed high byte
            speed_raw & 0xFF,              # Speed low byte
            (position_raw >> 8) & 0xFF,    # Position high byte
            position_raw & 0xFF,           # Position low byte
        ]
        return SimulatedMessage(arbitration_id=0x140 + self.motor_id, data=data)


# ──────────────────────────────────────────────
# Main CAN Bus Interface
# ──────────────────────────────────────────────

class CANBus:
    """
    Unified CAN bus interface supporting both simulated and real hardware.

    Parameters
    ----------
    simulated : bool
        If True, runs in simulation mode (no hardware needed).
        If False, connects to a real USB-to-CAN adapter via python-can.
    channel : str
        CAN channel name (ignored in simulation).
        Examples: 'COM3' (Windows), '/dev/ttyACM0' (Linux), 'PCAN_USBBUS1'
    bustype : str
        python-can bus type. Common values:
          'slcan'  - for CANable / cheap USB adapters (most common)
          'pcan'   - for PEAK PCAN-USB
          'kvaser' - for Kvaser hardware
          'socketcan' - for Linux SocketCAN (Raspberry Pi etc.)
    bitrate : int
        CAN bus speed in bits/sec. RMD-X8 default is 1,000,000 (1 Mbps).
    num_motors : int
        Number of motors in the system (used for simulation).
    """

    def __init__(
        self,
        simulated: bool = True,
        channel: str = "COM3",
        bustype: str = "slcan",
        bitrate: int = 1_000_000,
        num_motors: int = 6,
    ):
        self.simulated = simulated
        self.num_motors = num_motors
        self._bus = None
        self._lock = threading.Lock()

        # Simulated motor states (motor IDs 1–6)
        self._sim_motors = {
            i: SimulatedMotorState(i) for i in range(1, num_motors + 1)
        }
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_running = False

        if simulated:
            logger.info("CANBus: Starting in SIMULATION mode")
            self._start_simulation()
        else:
            logger.info(f"CANBus: Connecting to real hardware on {channel} ({bustype})")
            self._connect_real(channel, bustype, bitrate)

    # ── Simulation internals ──────────────────

    def _start_simulation(self):
        """Start the background thread that animates simulated motor physics."""
        self._sim_running = True
        self._sim_thread = threading.Thread(
            target=self._simulation_loop, daemon=True, name="SimMotorThread"
        )
        self._sim_thread.start()

    def _simulation_loop(self):
        dt = 0.005  # 5 ms physics step
        while self._sim_running:
            for motor in self._sim_motors.values():
                motor.step(dt)
            time.sleep(dt)

    # ── Real hardware connection ──────────────

    def _connect_real(self, channel: str, bustype: str, bitrate: int):
        """Connect to real USB-to-CAN hardware using python-can."""
        try:
            import can
            self._bus = can.interface.Bus(
                channel=channel,
                bustype=bustype,
                bitrate=bitrate,
            )
            logger.info("CANBus: Connected to real CAN hardware successfully.")
        except ImportError:
            raise ImportError(
                "python-can is not installed. Run: pip install python-can\n"
                "For CANable adapters also run: pip install python-can[serial]"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to CAN hardware on '{channel}': {e}\n"
                "Check that your USB-to-CAN adapter is plugged in and the channel is correct."
            )

    # ── Public send / receive API ─────────────

    def send(self, arbitration_id: int, data: list[int]) -> bool:
        """
        Send a CAN frame.

        Parameters
        ----------
        arbitration_id : int
            The CAN frame ID (e.g., 0x141 for motor 1).
        data : list[int]
            Up to 8 bytes of payload.

        Returns
        -------
        bool : True if sent successfully, False otherwise.
        """
        if self.simulated:
            # In simulation, parse the command and update motor target state
            self._sim_handle_command(arbitration_id, data)
            return True
        else:
            try:
                import can
                msg = can.Message(
                    arbitration_id=arbitration_id,
                    data=bytearray(data),
                    is_extended_id=False,
                )
                with self._lock:
                    self._bus.send(msg)
                return True
            except Exception as e:
                logger.error(f"CAN send error: {e}")
                return False

    def receive(self, timeout: float = 0.05) -> Optional[object]:
        """
        Receive a CAN frame.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for a message (seconds).

        Returns
        -------
        Message object or None if timeout.
        """
        if self.simulated:
            # Return the first motor's feedback as a demo
            # (In real use, feedback arrives naturally after each command)
            time.sleep(0.001)
            return None
        else:
            try:
                with self._lock:
                    msg = self._bus.recv(timeout=timeout)
                return msg
            except Exception as e:
                logger.error(f"CAN receive error: {e}")
                return None

    def get_simulated_state(self, motor_id: int) -> Optional[SimulatedMotorState]:
        """Returns live simulated motor state (simulation mode only)."""
        if self.simulated:
            return self._sim_motors.get(motor_id)
        return None

    # ── Simulation command parser ─────────────

    def _sim_handle_command(self, arb_id: int, data: list[int]):
        """Parse a command frame and update the simulated motor's target."""
        motor_id = arb_id - 0x140
        if motor_id not in self._sim_motors:
            return
        motor = self._sim_motors[motor_id]
        if not data:
            return

        cmd = data[0]

        # NOTE: No lock here — target writes are picked up by the physics thread
        # on its next iteration. The motor_driver sets target_position directly
        # in simulation mode (bypassing the CAN frame truncation), so this
        # handler only needs to handle velocity and stop commands.

        # Velocity control command (0xA2)
        if cmd == 0xA2 and len(data) >= 8:
            speed_raw = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
            if speed_raw > 0x7FFFFFFF:
                speed_raw -= 0x100000000
            motor.target_velocity = speed_raw * 0.01  # 0.01 dps/LSB
            motor.target_position = motor.position_deg  # hold position

        # Stop / shutdown command
        elif cmd in (0x81, 0x80):
            motor.target_position = motor.position_deg
            motor.target_velocity = 0.0

    # ── Cleanup ───────────────────────────────

    def close(self):
        """Release all resources."""
        self._sim_running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=1.0)
        if self._bus:
            self._bus.shutdown()
        logger.info("CANBus: Closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
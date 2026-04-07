"""
main.py
=======
Main Entry Point — MyActuator RMD-X8-120 Motor Control System
6-Motor Robotic Arm Test Suite

Author: Badhon Kumar
Date: April 2026

Run this file to test your 6-motor system.
Works in simulation mode by default (no hardware required).

Usage:
    python3.10 main.py                         # Simulation mode (default)
    python3.10 main.py --real --port COM3      # Real hardware on Windows
    python3.10 main.py --real --port /dev/ttyACM0 --bustype slcan  # Linux / Raspberry Pi

Hardware Setup Checklist (before switching to real mode):
    1. Install python-can:  pip install python-can python-can[serial]
    2. Connect USB-to-CAN adapter to PC
    3. Wire CAN-H and CAN-L from adapter to Motor 1 CAN-IN port
    4. Daisy-chain: Motor 1 CAN-OUT → Motor 2 CAN-IN → ... → Motor 6
    5. Attach 120Ω termination resistors at adapter end AND Motor 6 CAN-OUT
    6. Power up 48V power supply BEFORE enabling software
    7. Update --port to match your adapter (Device Manager on Windows, ls /dev/tty* on Linux)
"""

import argparse
import logging
import time
import sys

from can_interface import CANBus
from robot_controller import RobotController, DEFAULT_JOINT_CONFIG

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ══════════════════════════════════════════════
# TEST ROUTINES
# ══════════════════════════════════════════════

def test_0_connection(robot: RobotController):
    """
    TEST 0: Connection / Communication Check
    ─────────────────────────────────────────
    Reads status from each motor and prints it.
    Expected result: All 6 motors respond with position ≈ 0°.
    """
    print("\n" + "═" * 60)
    print("TEST 0: Connection & Communication Check")
    print("═" * 60)

    all_ok = True
    for cfg in robot.joint_configs:
        motor = robot.motors[cfg.motor_id]
        fb = motor.read_status()
        if fb:
            print(f"  ✓ Motor {cfg.motor_id} ({cfg.name:10s}): {fb}")
        else:
            print(f"  ✗ Motor {cfg.motor_id} ({cfg.name:10s}): NO RESPONSE")
            all_ok = False

    if all_ok:
        print("\n  ✓ All motors responding correctly.\n")
    else:
        print("\n  ✗ Some motors did not respond. Check wiring and IDs.\n")
    return all_ok


def test_1_single_joint(robot: RobotController):
    """
    TEST 1: Single Joint Movement
    ──────────────────────────────
    Moves each joint individually through a small range.
    Tests that each motor moves independently and correctly.
    """
    print("\n" + "═" * 60)
    print("TEST 1: Single Joint Movement (each joint individually)")
    print("═" * 60)

    test_positions = [30.0, -30.0, 0.0]   # degrees

    for cfg in robot.joint_configs:
        print(f"\n  Testing Joint {cfg.motor_id} — {cfg.name}")
        for target in test_positions:
            # Clamp to joint limits
            clamped = max(cfg.min_deg, min(cfg.max_deg, target))
            print(f"    → Moving to {clamped:+.0f}°", end="", flush=True)
            robot.move_joint(cfg.motor_id, clamped, speed_dps=90.0, wait=True, timeout_s=8.0)
            actual = robot.motors[cfg.motor_id].get_position()
            print(f"    Arrived at {actual:+.1f}°")
            time.sleep(0.5)

    print("\n  ✓ Single joint test complete.\n")


def test_2_synchronized_motion(robot: RobotController):
    """
    TEST 2: Synchronized Multi-Axis Motion
    ────────────────────────────────────────
    Moves all 6 joints simultaneously to a set of waypoints.
    This tests the core coordinated motion capability.
    """
    print("\n" + "═" * 60)
    print("TEST 2: Synchronized 6-Axis Motion")
    print("═" * 60)

    # Define a sequence of 6-joint poses: {motor_id: target_degrees}
    waypoints = [
        # Pose 1: Slight forward reach
        {1:  0.0, 2:  20.0, 3: -40.0, 4:  20.0, 5: 0.0, 6: 0.0},
        # Pose 2: Rotate base left, raise shoulder
        {1: 45.0, 2:  40.0, 3: -60.0, 4:  20.0, 5: 15.0, 6: 45.0},
        # Pose 3: Rotate base right
        {1:-45.0, 2:  20.0, 3: -30.0, 4: -10.0, 5:-15.0, 6:-45.0},
        # Pose 4: Return to home
        {1:  0.0, 2:   0.0, 3:   0.0, 4:   0.0, 5:  0.0, 6:  0.0},
    ]

    for i, pose in enumerate(waypoints, 1):
        print(f"\n  Waypoint {i}/{len(waypoints)}: Moving all joints...")
        start = time.time()
        robot.move_all_joints(pose, speed_dps=90.0, wait=True, timeout_s=15.0)
        elapsed = time.time() - start
        robot.print_status()
        print(f"    Motion completed in {elapsed:.2f}s")
        time.sleep(1.0)

    print("  ✓ Synchronized motion test complete.\n")


def test_3_velocity_control(robot: RobotController):
    """
    TEST 3: Velocity Control Mode
    ──────────────────────────────
    Spins Joint 6 (Tool) at constant velocity, then stops.
    Useful for end-effector or gripper control.
    """
    print("\n" + "═" * 60)
    print("TEST 3: Velocity Control (Joint 6 — Tool)")
    print("═" * 60)

    motor = robot.motors[6]

    print("  → Spinning at +120°/s for 2 seconds...")
    motor.set_velocity(120.0)
    time.sleep(2.0)

    print("  → Spinning at -120°/s for 2 seconds...")
    motor.set_velocity(-120.0)
    time.sleep(2.0)

    print("  → Stopping.")
    motor.set_velocity(0.0)
    time.sleep(0.5)
    motor.stop()

    print("  ✓ Velocity control test complete.\n")


def test_4_home_sequence(robot: RobotController):
    """
    TEST 4: Homing Sequence
    ────────────────────────
    Moves all joints to their defined home positions.
    Run this as the first and last step in any real session.
    """
    print("\n" + "═" * 60)
    print("TEST 4: Homing Sequence")
    print("═" * 60)

    print("  → Moving all joints to HOME (0° on all axes)...")
    success = robot.go_home(speed_dps=60.0, wait=True)

    if success:
        print("  ✓ All joints at home position.\n")
    else:
        print("  ✗ Homing timed out. Check motor limits and connections.\n")

    robot.print_status()
    return success


def test_5_real_time_feedback(robot: RobotController):
    """
    TEST 5: Real-Time Feedback Monitoring
    ──────────────────────────────────────
    Displays live feedback from all motors for 5 seconds.
    Demonstrates the monitoring loop running at ~50 Hz.
    """
    print("\n" + "═" * 60)
    print("TEST 5: Real-Time Feedback Monitoring (5 seconds)")
    print("═" * 60)

    start = time.time()
    while time.time() - start < 5.0:
        robot.print_status()
        time.sleep(1.0)

    print("  ✓ Feedback monitoring test complete.\n")


def test_6_emergency_stop(robot: RobotController):
    """
    TEST 6: Emergency Stop
    ───────────────────────
    Triggers e-stop mid-motion, verifying motors cut torque immediately.
    """
    print("\n" + "═" * 60)
    print("TEST 6: Emergency Stop Test")
    print("═" * 60)

    print("  → Starting motion on all joints...")
    # Send long motion commands (non-blocking)
    robot.move_all_joints(
        {1: 90.0, 2: 45.0, 3: -60.0, 4: 30.0, 5: 20.0, 6: 90.0},
        speed_dps=60.0,
        wait=False,
    )

    print("  → Waiting 1 second...")
    time.sleep(1.0)

    print("  → Triggering EMERGENCY STOP!")
    robot.estop()
    time.sleep(0.5)

    robot.print_status()
    print("  → E-stop confirmed. Resetting...")
    robot.reset()
    time.sleep(0.5)

    robot.go_home(speed_dps=60.0, wait=True)
    print("  ✓ Emergency stop test complete.\n")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

ALL_TESTS = {
    0: test_0_connection,
    1: test_1_single_joint,
    2: test_2_synchronized_motion,
    3: test_3_velocity_control,
    4: test_4_home_sequence,
    5: test_5_real_time_feedback,
    6: test_6_emergency_stop,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MyActuator RMD-X8-120 Motor Control — Test Suite"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Connect to real hardware (default: simulation mode)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="CAN adapter port (e.g., COM3, /dev/ttyACM0, PCAN_USBBUS1)",
    )
    parser.add_argument(
        "--bustype",
        type=str,
        default="slcan",
        choices=["slcan", "pcan", "kvaser", "socketcan", "canable"],
        help="python-can bus type (default: slcan for CANable adapters)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1_000_000,
        help="CAN bus bitrate in bps (default: 1000000 = 1 Mbps)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="0,4,2,3,5",
        help="Comma-separated test numbers to run (default: 0,4,2,3,5). "
             "Available: 0=connection, 1=single-joint, 2=sync-motion, "
             "3=velocity, 4=home, 5=feedback, 6=estop",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Print startup banner ──────────────────
    print("\n" + "═" * 60)
    print("  MyActuator RMD-X8-120 — 6-Motor Control System")
    print("  Power supply and Control System by Badhon Kumar")
    print("═" * 60)
    mode = "REAL HARDWARE" if args.real else "SIMULATION"
    print(f"  Mode    : {mode}")
    if args.real:
        print(f"  Port    : {args.port}")
        print(f"  Bus type: {args.bustype}")
    print(f"  Bitrate : {args.bitrate:,} bps")
    print("═" * 60 + "\n")

    # ── Parse test selection ──────────────────
    try:
        test_ids = [int(x.strip()) for x in args.tests.split(",")]
    except ValueError:
        print("Error: --tests must be comma-separated integers, e.g. '0,1,2'")
        sys.exit(1)

    # ── Create CAN bus and robot controller ───
    try:
        bus = CANBus(
            simulated=not args.real,
            channel=args.port,
            bustype=args.bustype,
            bitrate=args.bitrate,
            num_motors=6,
        )
    except (ImportError, ConnectionError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── Run selected tests ────────────────────
    with RobotController(bus, DEFAULT_JOINT_CONFIG) as robot:
        print("All motors enabled. Starting tests...\n")
        time.sleep(0.5)

        for test_id in test_ids:
            test_fn = ALL_TESTS.get(test_id)
            if test_fn is None:
                print(f"  [SKIP] Unknown test ID: {test_id}")
                continue
            try:
                test_fn(robot)
                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Keyboard interrupt — triggering E-STOP and shutting down.")
                robot.estop()
                break
            except Exception as e:
                logger.error(f"Test {test_id} failed with exception: {e}", exc_info=True)

        print("\n" + "═" * 60)
        print("  All selected tests complete. Homing before shutdown...")
        print("═" * 60)
        robot.go_home(speed_dps=60.0, wait=True)

    print("\n  System shutdown complete. Goodbye.\n")


if __name__ == "__main__":
    main()

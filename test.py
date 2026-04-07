"""
mission.py
==========
Write your custom robot movements here.
Run with: python3.10 mission.py
"""

import time
from can_interface import CANBus
from robot_controller import RobotController

def my_mission(robot: RobotController):

    # ── Step 1: Always home first ──────────────
    print("Homing...")
    robot.go_home(speed_dps=60.0, wait=True)
    time.sleep(1.0)

    # ── Step 2: Move individual joints ─────────
    # move_joint(motor_id, degrees, speed, wait)
    robot.move_joint(1, 45.0,  speed_dps=90.0,  wait=True)   # Base rotates 45°
    robot.move_joint(2, 30.0,  speed_dps=60.0,  wait=True)   # Shoulder up 30°
    time.sleep(0.5)

    # ── Step 3: Move all joints at once ────────
    robot.move_all_joints({
        1:  45.0,   # Base
        2:  30.0,   # Shoulder
        3: -60.0,   # Elbow
        4:  20.0,   # Wrist 1
        5: -10.0,   # Wrist 2
        6:   0.0,   # Tool
    }, speed_dps=90.0, wait=True)
    time.sleep(1.0)

    # ── Step 4: Spin the tool joint ────────────
    robot.motors[6].set_velocity(180.0)   # spin at 180°/s
    time.sleep(3.0)
    robot.motors[6].set_velocity(0.0)     # stop spinning
    time.sleep(0.5)

    # ── Step 5: Run a sequence of poses ────────
    poses = [
        {1:  0.0, 2:  10.0, 3: -20.0, 4:  0.0, 5:  0.0, 6:  0.0},
        {1: 30.0, 2:  20.0, 3: -40.0, 4: 10.0, 5: 10.0, 6: 90.0},
        {1: 60.0, 2:  30.0, 3: -60.0, 4: 20.0, 5: 20.0, 6: 180.0},
        {1:  0.0, 2:   0.0, 3:   0.0, 4:  0.0, 5:  0.0, 6:  0.0},  # back to home
    ]

    for i, pose in enumerate(poses, 1):
        print(f"Moving to pose {i}...")
        robot.move_all_joints(pose, speed_dps=90.0, wait=True)
        time.sleep(0.5)

    # ── Step 6: Always home at the end ─────────
    print("Returning home...")
    robot.go_home(speed_dps=60.0, wait=True)


# ── Entry point ────────────────────────────────
if __name__ == "__main__":
    bus = CANBus(simulated=True)   # change to False for real hardware

    with RobotController(bus) as robot:
        my_mission(robot)
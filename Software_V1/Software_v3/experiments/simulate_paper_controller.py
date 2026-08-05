"""Run a hardware-free smoke simulation of the V3 paper controller."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from Software_v3.config import PaperParameters
from Software_v3.control import JacobianController
from Software_v3.experiments.trajectories import trajectory
from Software_v3.io import LowPassFilterBank
from Software_v3.model import forward_kinematics


def _metrics(errors: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rmse": np.sqrt(np.mean(errors**2, axis=0)),
        "mae": np.mean(np.abs(errors), axis=0),
        "maxae": np.max(np.abs(errors), axis=0),
    }


def run_simulation(
    trajectory_name: str = "traj3",
    period_s: float = 40.0,
    duration_s: float | None = None,
    use_kalman: bool = True,
    output_csv: Path | None = None,
) -> dict[str, np.ndarray]:
    params = PaperParameters()
    dt = 1.0 / params.control_hz
    duration_s = float(duration_s if duration_s is not None else period_s)
    ref_fn = trajectory(trajectory_name, period_s)

    controller = JacobianController(params=params, use_kalman=use_kalman)
    pose_filter = LowPassFilterBank(params.filter_cutoff_hz, params.control_hz, channels=3)

    u = np.zeros(3, dtype=float)
    rows: list[dict[str, float]] = []
    errors: list[np.ndarray] = []

    # Simulation plant: exact PCC for now. The estimator/controller are still
    # exercised, and future tests can replace this with measured pose feedback.
    steps = int(round(duration_s / dt))
    for step in range(steps + 1):
        t = step * dt
        pose_ref = ref_fn(t)
        pose_measured = forward_kinematics(
            u,
            params.segment_lengths_m,
            params.tendon_offsets_m,
        )

        pose_for_estimator = pose_filter.update(pose_measured)
        control = controller.step(u, pose_ref, pose_for_estimator)
        u = control.u_next

        err = pose_ref - pose_measured
        err[2] = (err[2] + np.pi) % (2.0 * np.pi) - np.pi
        errors.append(err)

        rows.append(
            {
                "t_s": t,
                "x_ref_m": pose_ref[0],
                "y_ref_m": pose_ref[1],
                "psi_ref_rad": pose_ref[2],
                "x_m": pose_measured[0],
                "y_m": pose_measured[1],
                "psi_rad": pose_measured[2],
                "err_x_m": err[0],
                "err_y_m": err[1],
                "err_psi_rad": err[2],
                "u1_m": u[0],
                "u2_m": u[1],
                "u3_m": u[2],
                "innovation_norm": float(np.linalg.norm(control.innovation)),
                "delta_j_norm": float(np.linalg.norm(control.delta_jacobian)),
            }
        )

    error_array = np.vstack(errors)
    result = _metrics(error_array)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default="traj3", choices=["traj1", "traj2", "traj3"])
    parser.add_argument("--period", type=float, default=40.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--pcc-only", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    result = run_simulation(
        trajectory_name=args.trajectory,
        period_s=args.period,
        duration_s=args.duration,
        use_kalman=not args.pcc_only,
        output_csv=args.csv,
    )

    rmse = result["rmse"]
    mae = result["mae"]
    maxae = result["maxae"]
    print(f"Trajectory: {args.trajectory}, T={args.period:g}s, method={'PCC only' if args.pcc_only else 'PCC + Kalman'}")
    print(f"RMSE:  x={rmse[0]*1000:.3f} mm, y={rmse[1]*1000:.3f} mm, psi={np.rad2deg(rmse[2]):.3f} deg")
    print(f"MAE:   x={mae[0]*1000:.3f} mm, y={mae[1]*1000:.3f} mm, psi={np.rad2deg(mae[2]):.3f} deg")
    print(f"MaxAE: x={maxae[0]*1000:.3f} mm, y={maxae[1]*1000:.3f} mm, psi={np.rad2deg(maxae[2]):.3f} deg")


if __name__ == "__main__":
    main()

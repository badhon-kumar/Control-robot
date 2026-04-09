from single_input_cartesian import parse_csv
import os
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartesian_sequence.csv")
steps, err = parse_csv(csv_path)
print(f"{len(steps)} steps loaded")
for s in steps:
    motors = s.motors_to_move()
    parts = ", ".join(f"Motor{m} -> {a}deg @{sp}dps" for m, a, sp in motors)
    print(f"  [{s.step_label}] delay={s.delay_s}s  {parts or '(delay only)'}")
if err:
    print(f"Warnings: {err}")
else:
    print("No errors.")

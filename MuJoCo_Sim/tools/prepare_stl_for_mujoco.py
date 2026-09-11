"""
Convert an STL to an aligned binary STL for MuJoCo.

The source CAD files are in millimetres and may not be centered at the joint
origin. For the both_side disk, this script can align the proximal pin hole to
x = 0 and scale the distal pin hole to a target pitch, so consecutive STL disks
share the same revolute-pin locations in MuJoCo.

Run:
    python tools/prepare_stl_for_mujoco.py assets/both_side.STL assets/both_side_mujoco.stl --pin-pitch 15
    python tools/prepare_stl_for_mujoco.py assets/one_side.STL assets/one_side_mujoco.stl --pin-origin 3.6710065 6.5 6.5
"""

from pathlib import Path
import argparse
import struct


def read_stl(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50 * n == len(raw):
            facets = []
            off = 84
            for _ in range(n):
                nums = struct.unpack("<12fH", raw[off:off + 50])[:12]
                off += 50
                facets.append((nums[0:3], nums[3:6], nums[6:9], nums[9:12]))
            return facets

    facets = []
    normal = None
    verts = []
    for line in raw.decode(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 5 and parts[0] == "facet" and parts[1] == "normal":
            normal = tuple(float(x) for x in parts[2:5])
            verts = []
        elif len(parts) == 4 and parts[0] == "vertex":
            verts.append(tuple(float(x) for x in parts[1:4]))
        elif parts[:1] == ["endfacet"] and normal is not None and len(verts) == 3:
            facets.append((normal, verts[0], verts[1], verts[2]))
    if not facets:
        raise SystemExit(f"No STL facets found in {path}")
    return facets


def bounds(facets):
    verts = [p for _, a, b, c in facets for p in (a, b, c)]
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    return lo, hi


def transformed(facets, pitch_mm=None, pin_origin=None, axis_order="xyz",
                uniform_scale=1.0):
    lo, hi = bounds(facets)
    if pin_origin is not None:
        ox, oy, oz = pin_origin

        def shift_pin(p):
            d = {"x": p[0] - ox, "y": p[1] - oy, "z": p[2] - oz}
            return tuple(d[c] * uniform_scale for c in axis_order)

        return [(n, shift_pin(a), shift_pin(b), shift_pin(c))
                for n, a, b, c in facets], lo, hi

    if pitch_mm is None:
        ctr = [(lo[i] + hi[i]) / 2.0 for i in range(3)]

        def shift(p):
            return tuple(p[i] - ctr[i] for i in range(3))

        return [(n, shift(a), shift(b), shift(c)) for n, a, b, c in facets], lo, hi

    # Measured from both_side.STL: the circular pin-hole rims in x-y projection.
    prox_x = 0.99201295
    distal_x = 6.85242550
    center_y = 6.5
    center_z = 6.5
    sx = float(pitch_mm) / (distal_x - prox_x)

    def align(p):
        return ((p[0] - prox_x) * sx, p[1] - center_y, p[2] - center_z)

    return [(n, align(a), align(b), align(c)) for n, a, b, c in facets], lo, hi


def write_binary_stl(path, facets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"mujoco_centered_binary_stl".ljust(80, b" ")
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(facets)))
        for n, a, b, c in facets:
            f.write(struct.pack(
                "<12fH",
                n[0], n[1], n[2],
                a[0], a[1], a[2],
                b[0], b[1], b[2],
                c[0], c[1], c[2],
                0,
            ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--pin-pitch", type=float, default=None,
                    help="target proximal-to-distal pin-hole pitch in mm")
    ap.add_argument("--pin-origin", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="source STL pin-hole center to place at output origin, in mm")
    ap.add_argument("--axis-order", default="xyz", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"),
                    help="output axis mapping from source axes")
    ap.add_argument("--uniform-scale", type=float, default=1.0,
                    help="uniform scale applied before STL mm -> MuJoCo m scale")
    args = ap.parse_args()

    facets = read_stl(args.source)
    facets, lo, hi = transformed(facets, args.pin_pitch, args.pin_origin,
                                 args.axis_order, args.uniform_scale)
    write_binary_stl(args.output, facets)
    size = [hi[i] - lo[i] for i in range(3)]
    print(f"Wrote {args.output}")
    print(f"Facets: {len(facets)}")
    print(f"Original bounds min: {lo}")
    print(f"Original bounds max: {hi}")
    print(f"Original size mm: {size}")
    if args.pin_origin is not None:
        print(f"Output units: millimetres, pin origin shifted from {args.pin_origin} to 0,0,0")
    elif args.pin_pitch is None:
        print("Output units: millimetres, centered at bounding-box center")
    else:
        print(f"Output units: millimetres, proximal pin at x=0, distal pin at x={args.pin_pitch:g}")


if __name__ == "__main__":
    main()

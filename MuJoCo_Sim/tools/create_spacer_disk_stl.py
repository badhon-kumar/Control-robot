"""
Generate an estimated spacer disk STL for the Zhai et al. continuum robot.

Units in the STL are millimetres. Coordinate convention:
  x = backbone / robot length direction
  y = bending-plane lateral direction
  z = revolute pin axis

This is an estimated visual/CAD starting point from the paper dimensions:
13 mm manipulator diameter, 1 mm NiTi revolute pins, and raised lug features
estimated from Fig. 2B. The paper does not publish full lug CAD dimensions.

Run:
    python tools/create_spacer_disk_stl.py
"""

from pathlib import Path
import math
import struct

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "spacer_disk_estimated.stl"


DIMS = {
    "disk_radius": 6.5,
    "disk_thickness": 1.5,
    "lug_length_x": 3.5,
    "lug_width_y": 1.6,
    "lug_height_z": 4.0,
    "lug_y_offset": 4.45,
    "pin_clearance_diameter": 1.2,
}


def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    L = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / L, ny / L, nz / L


def tri(facets, a, b, c):
    facets.append((normal(a, b, c), a, b, c))


def cylinder_x(facets, name, radius, x0, x1, n=128):
    # Cylinder axis is x. Closed with fan caps.
    c0 = (x0, 0.0, 0.0)
    c1 = (x1, 0.0, 0.0)
    for i in range(n):
        a0 = 2.0 * math.pi * i / n
        a1 = 2.0 * math.pi * (i + 1) / n
        p0 = (x0, radius * math.cos(a0), radius * math.sin(a0))
        p1 = (x0, radius * math.cos(a1), radius * math.sin(a1))
        p2 = (x1, radius * math.cos(a1), radius * math.sin(a1))
        p3 = (x1, radius * math.cos(a0), radius * math.sin(a0))
        tri(facets, p0, p1, p2)
        tri(facets, p0, p2, p3)
        tri(facets, c0, p1, p0)
        tri(facets, c1, p3, p2)


def ray_to_rect(cx, cy, dx, dy, hx, hy):
    tx = hx / abs(dx) if abs(dx) > 1e-12 else float("inf")
    ty = hy / abs(dy) if abs(dy) > 1e-12 else float("inf")
    t = min(tx, ty)
    return cx + t * dx, cy + t * dy


def lug_with_pin_hole(facets, cx, cy, cz, sx, sy, sz, hole_r, n=80):
    """
    Rectangular lug block with a through-hole along z.

    The top and bottom faces are meshed as a star-shaped annulus from the hole
    circle to the rectangle boundary; side faces are plain rectangle walls.
    """
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    z0, z1 = cz - hz, cz + hz
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy

    # Four outer side walls.
    corners = [
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),
        ((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)),
        ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)),
    ]
    for a, b, c, d in corners:
        tri(facets, a, b, c)
        tri(facets, a, c, d)

    # Top/bottom annular faces and inner cylindrical wall.
    for i in range(n):
        a0 = 2.0 * math.pi * i / n
        a1 = 2.0 * math.pi * (i + 1) / n
        d0 = (math.cos(a0), math.sin(a0))
        d1 = (math.cos(a1), math.sin(a1))

        h0 = (cx + hole_r * d0[0], cy + hole_r * d0[1])
        h1 = (cx + hole_r * d1[0], cy + hole_r * d1[1])
        o0 = ray_to_rect(cx, cy, d0[0], d0[1], hx, hy)
        o1 = ray_to_rect(cx, cy, d1[0], d1[1], hx, hy)

        hb0, hb1 = (h0[0], h0[1], z0), (h1[0], h1[1], z0)
        ob0, ob1 = (o0[0], o0[1], z0), (o1[0], o1[1], z0)
        ht0, ht1 = (h0[0], h0[1], z1), (h1[0], h1[1], z1)
        ot0, ot1 = (o0[0], o0[1], z1), (o1[0], o1[1], z1)

        tri(facets, ob0, hb0, hb1)
        tri(facets, ob0, hb1, ob1)
        tri(facets, ot0, ht1, ht0)
        tri(facets, ot0, ot1, ht1)

        tri(facets, hb0, ht0, ht1)
        tri(facets, hb0, ht1, hb1)


def write_binary_stl(path, facets):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"spacer_disk_estimated_mm_binary".ljust(80, b" ")
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
    d = DIMS
    facets = []

    cylinder_x(
        facets,
        "disk",
        d["disk_radius"],
        -d["disk_thickness"] / 2.0,
        d["disk_thickness"] / 2.0,
    )

    lug_cx = d["disk_thickness"] / 2.0 + d["lug_length_x"] / 2.0
    for sign in (-1.0, 1.0):
        lug_with_pin_hole(
            facets,
            cx=lug_cx,
            cy=sign * d["lug_y_offset"],
            cz=0.0,
            sx=d["lug_length_x"],
            sy=d["lug_width_y"],
            sz=d["lug_height_z"],
            hole_r=d["pin_clearance_diameter"] / 2.0,
        )

    write_binary_stl(OUT, facets)
    print(f"Wrote {OUT}")
    print(f"Facets: {len(facets)}")
    print("Units: millimetres")


if __name__ == "__main__":
    main()

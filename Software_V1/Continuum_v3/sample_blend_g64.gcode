; The square path from sample_square_path.gcode, with rounded corners.
;
; G64 P<tol>  allows the path to cut each corner by up to <tol> mm so the
;             reference never has to change direction instantly.
; G61         turns blending back off (exact stop, sharp corners).
;
; Raise P for smoother motion, lower it to follow the corners more exactly.
; Try P0.2 and P2.0 and watch the corners in the trajectory plot.

G17 G21 G90
F400

G64 P1.0            ; round corners by up to 1 mm

G0 X260 Y0 A0
G1 X260 Y40 A0
G1 X220 Y40 A0
G1 X220 Y0 A0
G1 X260 Y0 A0
G4 P1

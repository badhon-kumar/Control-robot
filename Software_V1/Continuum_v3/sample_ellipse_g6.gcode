; Paper Traj. 1 (Zhai et al. 2025, Fig. 6) as a single G6 ellipse.
;   xr = 240 + 20 cos(2*pi*t/T) mm
;   yr =       60 sin(2*pi*t/T) mm
;   psi = 0 (horizontal attitude held)
;
; Perimeter is ~267 mm, so F400 (= 6.67 mm/s) completes one lap in ~40 s,
; matching the paper's baseline period T = 40 s.
;
; Compare with sample_ellipse_path.gcode, which approximates a different
; ellipse using 20 hand-typed G1 chords.

G17 G21 G90         ; XY plane, millimetres, absolute
F400                ; 400 mm/min -> one lap in ~40 s

G0 X260 Y0 A0       ; move to the ellipse start without tracing
G6 X240 Y0 I20 J60  ; centre (240, 0), X semi-radius 20, Y semi-radius 60
G4 P1               ; dwell 1 s

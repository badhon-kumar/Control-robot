; Arcs written in the R (radius) form instead of I/J centre offsets.
; This is what most CAD/CAM software emits, so files can now be used unedited.
;
;   G2 = clockwise, G3 = counter-clockwise
;   R > 0 -> the short way round (sweep <= 180 deg)
;   R < 0 -> the long way round  (sweep >  180 deg)
;
; A full circle cannot be written with R (start and end coincide, so the radius
; is ambiguous) — use the I/J form for that, as in sample_circle_path.gcode.

G17 G21 G90
F400

G0 X260 Y0 A0
G3 X240 Y20 R20 A0      ; quarter turn, counter-clockwise
G3 X220 Y0 R20 A0       ; quarter turn, counter-clockwise
G2 X240 Y-20 R20 A0     ; quarter turn back, clockwise
G2 X260 Y0 R20 A0       ; quarter turn back, clockwise
G4 P1

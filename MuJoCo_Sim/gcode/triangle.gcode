; Continuum robot G-code sample: equilateral triangle
; Units are millimetres. X/Y are desired end-effector position.
; A is desired end-effector attitude in degrees.
;
; Side length 40 mm, so height = 40 * sqrt(3)/2 = 34.64 mm.
; Base runs X220..X260 at Y0, apex at X240 Y34.64 - the same
; workspace box as square.gcode, so the shapes are comparable.
;
; Three sharp corners make this the hardest of the four paths to
; track: each corner demands an instant change of direction.

G21         ; millimetres
G90         ; absolute coordinates
F600        ; 600 mm/min = 10 mm/s

G0 X220 Y0 A0
G1 X260 Y0 A0
G1 X240 Y34.64 A0
G1 X220 Y0 A0
G4 P1       ; dwell for 1 second

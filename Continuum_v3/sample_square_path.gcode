; Simple continuum robot G-code sample: square path
; Units are millimetres. X/Y are desired end-effector position.
; A is desired end-effector attitude in degrees.

G21         ; millimetres
G90         ; absolute coordinates
F600        ; 600 mm/min = 10 mm/s

G0 X260 Y0 A0
G1 X260 Y40 A0
G1 X220 Y40 A0
G1 X220 Y0 A0
G1 X260 Y0 A0
G4 P1       ; dwell for 1 second

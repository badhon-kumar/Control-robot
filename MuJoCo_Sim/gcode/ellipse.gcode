; Simple continuum robot G-code sample: ellipse path
; Units are millimetres. X/Y are desired end-effector position.
; A is desired end-effector attitude in degrees.
;
; This ellipse is approximated with short G1 line segments.
; Centre: X240 Y20, radius X20, radius Y15.

G21         ; millimetres
G90         ; absolute coordinates
F500        ; 500 mm/min

G0 X260.0 Y20.0 A0
G1 X259.0 Y24.6 A0
G1 X256.2 Y28.8 A0
G1 X251.8 Y32.1 A0
G1 X246.2 Y34.3 A0
G1 X240.0 Y35.0 A0
G1 X233.8 Y34.3 A0
G1 X228.2 Y32.1 A0
G1 X223.8 Y28.8 A0
G1 X221.0 Y24.6 A0
G1 X220.0 Y20.0 A0
G1 X221.0 Y15.4 A0
G1 X223.8 Y11.2 A0
G1 X228.2 Y7.9 A0
G1 X233.8 Y5.7 A0
G1 X240.0 Y5.0 A0
G1 X246.2 Y5.7 A0
G1 X251.8 Y7.9 A0
G1 X256.2 Y11.2 A0
G1 X259.0 Y15.4 A0
G1 X260.0 Y20.0 A0
G4 P1       ; dwell for 1 second

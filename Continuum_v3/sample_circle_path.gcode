; Simple circular path using G2/G3 arcs
; Starts at right side of circle and traces a full circle in two half-arcs.

G21         ; millimetres
G90         ; absolute coordinates
F500        ; 500 mm/min

G0 X260 Y20 A0
G3 X220 Y20 I-20 J0 A0
G3 X260 Y20 I20 J0 A0
G4 P1

; Free-form S-curve using cubic Bezier splines (G5).
;
; G5 I<dx> J<dy> P<dx> Q<dy> X<end> Y<end>
;   I/J = first control point, as an offset from the CURRENT point
;   P/Q = second control point, as an offset from the END point
;
; A following G5 may omit I and J: the control point is then mirrored from the
; previous curve, which makes the join tangent-continuous (no kink).
;
; Attitude uses A. Inside a G5, P and Q are curve geometry, never attitude.

G17 G21 G90
F400
G64 P0.5            ; round every join by up to 0.5 mm (modal: applies from here on)

G0 X245 Y-50 A0

; Lower half of the S: bulges right, then left.
G5 I15 J15 P-15 Q-15 X245 Y0 A0

; Upper half: I/J omitted, so it continues smoothly out of the curve above.
G5 P15 Q-15 X245 Y50 A0

; Return along two straight lines; the joins are rounded by the G64 above.
G1 X255 Y0 A0
G1 X245 Y-50 A0
G4 P1

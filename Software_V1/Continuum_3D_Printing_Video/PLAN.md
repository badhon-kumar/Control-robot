# Soft Continuum Arm 3D Printing Visualization Workflow

## Project Objective

Create a professional simulation video that demonstrates how the soft continuum arm can be used as a 3D-printing manipulator by attaching a nozzle to the end-effector and making the tip follow a print path while leaving a deposited material trail.

The initial target is a clear proof-of-concept visualization, not a complete physical printing controller.

## Demonstration Concept

The simulated continuum arm is extended from path tracking to additive manufacturing visualization by:

1. Treating the robot tip as a nozzle/toolhead.
2. Loading or generating a printing path, preferably from G-code.
3. Moving the continuum arm tip along that path.
4. Showing deposited material behind the nozzle.
5. Extending the path layer by layer to mimic 3D printing.

## Proposed Project Structure

```text
Continuum_3D_Printing_Video/
├── PLAN.md
├── README.md                         # Later: how to run the video generator
├── generate_print_video.py            # Main script for video creation
├── soft_arm_model.py                  # Continuum arm geometry / visualization helpers
├── gcode_path.py                      # G-code or path generation utilities
├── samples/
│   ├── square_print.gcode
│   ├── ellipse_print.gcode
│   └── layered_box.gcode
├── outputs/
│   ├── frames/
│   └── soft_arm_3d_printing_demo.mp4
└── assets/
    └── optional visual assets
```

## Implementation Workflow

### Phase 1: Proof-Of-Concept Animation

Objective: produce a working animation that clearly communicates the core concept.

Tasks:

1. Create a simple 3D scene:
   - print bed,
   - coordinate axes,
   - continuum arm backbone,
   - end-effector nozzle,
   - reference print path.

2. Generate a simple path:
   - square,
   - circle,
   - ellipse,
   - or imported G-code path.

3. Move the nozzle along the path:
   - compute the desired tip position at each animation frame,
   - draw the continuum arm reaching that point,
   - keep the nozzle attached to the arm tip.

4. Draw deposited material:
   - every previous tip position becomes part of the printed trail,
   - use a thick colored line to look like extruded filament.

5. Export video:
   - save as `.mp4`,
   - use a stable camera angle,
   - include title and frame labels only if useful.

Recommended tool: `matplotlib` 3D animation first because it is simple and reliable.

### Phase 2: Visual Refinement

Objective: improve the visual quality so the video is suitable for technical presentation.

Improvements:

1. Use a smoother arm body:
   - draw the continuum arm as a thick tube instead of a thin line,
   - show multiple backbone points,
   - optionally color each segment differently.

2. Add a nozzle model:
   - small cone or cylinder at the tip,
   - dark metal color,
   - slight orientation aligned with the final arm direction.

3. Improve deposited filament:
   - thicker line,
   - rounded appearance,
   - material color such as orange, blue, or green,
   - optional layer height effect.

4. Improve camera:
   - use an isometric 3D view,
   - keep the arm and print bed fully visible,
   - avoid excessive zooming or shaking.

5. Add path preview:
   - show unprinted path as a faint dashed line,
   - show printed path as a solid bright line.

### Phase 3: G-Code-Based Print Path Demonstration

Objective: connect the visualization workflow to the existing `Continuum_v3` G-code path-tracking work.

Tasks:

1. Reuse or copy the G-code parsing idea from `Continuum_v3`.
2. Support useful commands:
   - `G0` rapid movement,
   - `G1` straight printing movement,
   - `G2/G3` circular arcs,
   - `G21` millimetres,
   - `G90` absolute positioning,
   - optional `F` feedrate.

3. Add extrusion-like behavior:
   - printing occurs during `G1/G2/G3`,
   - no material is deposited during `G0`,
   - optional future support for `E` extrusion values.

4. Create sample files:
   - square path,
   - ellipse approximation,
   - circle path,
   - simple layered shape.

### Phase 4: Layered Printing Visualization

Objective: extend the demonstration from 2D path tracing to a layered additive manufacturing process.

Tasks:

1. Add Z/layer height:
   - first layer at `Z=0`,
   - second layer at `Z=1` or similar,
   - repeat same or slightly modified path.

2. Show multiple printed layers:
   - previous layers remain visible,
   - current layer is highlighted,
   - nozzle moves above the currently printed layer.

3. Add a basic print object:
   - square wall,
   - ellipse wall,
   - small rounded rectangle,
   - simple vase-like outline.

### Phase 5: Presentation-Ready Export

Objective: generate a video suitable for PowerPoint, project documentation, and technical discussion.

Output requirements:

1. Video format:
   - `.mp4`,
   - 1080p if possible,
   - 20-30 FPS,
   - 20-40 seconds duration.

2. Optional still images:
   - title frame,
   - final printed shape,
   - frame showing nozzle and deposited material.

3. Presentation-friendly labels:
   - "Soft continuum arm",
   - "Nozzle/end-effector",
   - "G-code path",
   - "Deposited filament".

Keep labels minimal so the video remains clean.

## Recommended Initial Demonstration

Start with a layered ellipse or square because it is easy to understand visually.

Suggested video sequence:

1. Show print bed and target path.
2. Show continuum arm moving from rest position.
3. Attach nozzle at tip.
4. Nozzle follows the first layer path.
5. Filament trail appears behind nozzle.
6. Nozzle lifts slightly and prints second layer.
7. Final printed two-layer shape remains on bed.

## Technical Methodology

### Arm Visualization

Use the current continuum arm model as inspiration, but the video does not need the full controller at first. For the first video, the arm can be drawn as a smooth curve from base to desired nozzle position.

Possible approach:

1. Define base point.
2. Define desired tip/nozzle point from the print path.
3. Generate a smooth backbone curve between base and tip.
4. Draw the curve as the soft arm.
5. Place nozzle at the final point.

Later, the real forward kinematics from `Continuum_v3` can be connected for more realistic arm shape.

### Print Path

Use millimetres as in the existing simulation:

```text
X = print bed horizontal direction
Y = print bed depth direction
Z = layer height
```

The existing `Continuum_v3` paths are mostly X-Y. For video printing, we can treat them as print-bed paths and add a fixed or changing Z height.

### Material Deposition

Maintain a list of printed points:

```text
printed_points = all previous nozzle positions where printing is active
```

During each frame:

1. Move nozzle to current point.
2. If command is printing, append current point to printed trail.
3. Draw all printed points as filament.

## Presentation Enhancements

Recommended enhancements after the first working video:

1. Smooth camera rotation at the beginning or end.
2. Clean color scheme:
   - dark grey print bed,
   - blue/teal soft arm,
   - metallic nozzle,
   - orange deposited filament.
3. Small shadow or grid on print bed.
4. Faint reference path before printing.
5. Export both a short version and a slower explanation version.

## Current Technical Limitations

The initial video will be a visualization/proof-of-concept. It will not yet validate real printing performance.

Limitations:

1. No real extrusion pressure or flow model.
2. No nozzle-material interaction model.
3. No collision detection.
4. No closed-loop camera feedback.
5. No real hardware calibration.
6. No compensation for soft arm deformation under nozzle load.
7. No thermal/material curing model.

## Future Development Directions

After the first video is complete, possible next improvements include:

1. Add true G-code `E` extrusion support.
2. Add nozzle orientation control.
3. Add layered slicing from simple geometry.
4. Add physical constraints of the soft arm workspace.
5. Add collision checking with the print bed and printed object.
6. Use real forward kinematics from the continuum model.
7. Use camera or sensor feedback for real nozzle position.
8. Convert video simulation into a hardware-ready planning module.

## Completion Criteria

The result should clearly show:

1. A soft continuum arm with a nozzle at the tip.
2. The nozzle following a planned path.
3. Material being deposited along the path.
4. The final printed shape visible on the bed.
5. The video exported in a format that can be inserted into PowerPoint.

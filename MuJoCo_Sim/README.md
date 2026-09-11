# MuJoCo_Sim - planar continuum manipulator simulation

**Author:** Badhon Kumar

Self-contained MuJoCo simulation of the 3-segment tendon-driven continuum arm.
The important paper workflow is now one command family:

```powershell
cd "D:\Abroad\Meeting\Prof Cao\Robotics Control system\MuJoCo_Sim"
python run.py figures
python run.py fig6
python run.py fig7
python run.py fig12
python run.py fig6 --real
python run.py fig6 --real --save
```

## Paper Figures

`python run.py fig6`, `fig7`, or `fig12` opens the MuJoCo viewer. Nothing
is saved just by opening it. For tracking figures, `--save` runs the headless
Proposed/PCC comparison and writes the figures directly.

Viewer keys:

| Key | Action |
| :-- | :-- |
| `S` | start / stop the run |
| `P` | Fig. 12 only: attach the selected payload now |
| `G` | save completed run data and figures |
| `R` | restart the current choice |
| `K` | Fig. 6/7 only: switch Proposed / PCC model |
| `1` `2` `3` | Fig. 12 only: choose payload mass |
| `T` | show / hide trail |
| `[` `]` | slower / faster |

Saving writes to a fixed folder and clears the previous contents first:

```text
outputs/fig6/
outputs/fig7/
outputs/fig12/
```

So if you run the same figure multiple times, only the last saved result remains.
If you do not finish a run and press `G`, no data or figure is generated.

For Fig. 12, press `S` first and let the arm reach the fixed pose. Then press
`P` to attach the selected payload; the controller will compensate from there.
Panels D/E/F are the three payloads, so run payload `1`, `2`, and `3` in the
same viewer session, then press `G`. If you only run one payload, only that
payload is saved.

## Clean Structure

The paper workflow has three files:

| File | Job |
| :-- | :-- |
| `continuum_sim/experiments.py` | what Fig. 6, Fig. 7, and Fig. 12 are |
| `tools/paper_viewer.py` | MuJoCo visualization and run recording |
| `continuum_sim/figures.py` | saved plots, CSV files, and results text |

To change duration, control timestep, reference trajectory, gravity, or payload
masses, edit `continuum_sim/experiments.py`.

## Other Commands

```powershell
python run.py check all
python run.py inspect
python run.py live
python run.py build
python run.py sync
```

`check all` verifies the simulation setup. `inspect` is a manual-drive viewer.
`live` is the general G-code tracking demo. `paper` is the clean workflow for
your Fig. 6, Fig. 7, and Fig. 12 tasks.

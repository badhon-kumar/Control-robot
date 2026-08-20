"""
Planar continuum-manipulator simulation: the MuJoCo plant and its support code.

The plant is deliberately independent of the controller. It is built from
params.py alone and shares no model with continuum_ellipse.py, so the tracking
error it produces is a real measurement rather than a model agreeing with itself.

    params      geometry and material constants - the single source of truth
    build_model MJCF generator; models/continuum_planar.xml is its output
    plant       ContinuumPlant: step(u) -> tip pose [x, y, psi] in SI
    bridge      closes the loop between a controller and the plant
    calibrate   sweeps, model-error statistics, stiffness fitting
    paths       every directory, defined once
    vendor      access to and integrity checking of controller/

Author: Badhon Kumar
"""

from . import paths

__all__ = ["paths"]

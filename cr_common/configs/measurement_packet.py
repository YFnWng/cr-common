"""MeasurementPacket: typed container for one timestep's sensor observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class MeasurementPacket:
    """Typed container for one timestep's sensor observations.

    All position/pose values are in the world frame, in mm (positions) and
    radians (orientations).

    Attributes
    ----------
    timestamp : float
        Simulation or wall-clock time (s).
    dt : float
        Duration of this timestep (s).
    base_pose : np.ndarray, shape (7,), optional
        Base frame pose [x, y, z, qx, qy, qz, qw].  Required when
        ``known_base_pose=True`` in the estimator; may be ``None`` otherwise.
    cable_disp : float, optional
        Cable displacement (mm).  Set when cable is displacement-controlled.
    cable_tension : float, optional
        Cable tension (N).  Set when cable is force-controlled, or ``None``
        to let the estimator treat cable tension as a free variable.
    positions : dict[int, np.ndarray]
        node_index → (3,) observed position.  From MRI coils or EM position.
    poses : dict[int, np.ndarray]
        node_index → (7,) observed pose [x,y,z,qx,qy,qz,qw].  From EM coils.
    strains : dict[int, np.ndarray]
        section_index → (3,) observed midpoint curvature [u1,u2,u3].  From FBG.
        Becomes a midpoint factor: ``(S(k)+S(k+1))/2 = κ_obs[k]``.
    node_strains : dict[int, np.ndarray]
        node_index → (3,) observed curvature [u1,u2,u3].  Direct node prior.
        Becomes a ``PriorFactorVector(S(k), κ_obs)``.  Useful when a sensor
        directly measures curvature at a specific backbone node rather than
        a section midpoint.
    wrenches : dict[int, np.ndarray]
        node_index → (6,) observed wrench [moment; force].  From F/T sensors.
    base_commands : np.ndarray, shape (n_base_commands,), optional
        Proximal actuator commands u_ext (e.g. insertion translation, base
        rotation).  Used by the proximal-boundary factor as the ``u_ext``
        portion of ``u_act = [Q; u_ext]``.  ``None`` means the proximal
        boundary factor is not used or there are no external base commands.
    """

    timestamp: float
    dt: float
    base_pose: Optional[np.ndarray]
    cable_disp: Optional[float] = None
    cable_tension: Optional[float] = None
    cable_tensions: Optional[np.ndarray] = None  # shape (n_cables,); preferred for multi-cable
    positions: Dict[int, np.ndarray] = field(default_factory=dict)
    poses: Dict[int, np.ndarray] = field(default_factory=dict)
    strains: Dict[int, np.ndarray] = field(default_factory=dict)
    node_strains: Dict[int, np.ndarray] = field(default_factory=dict)
    wrenches: Dict[int, np.ndarray] = field(default_factory=dict)
    base_commands: Optional[np.ndarray] = None

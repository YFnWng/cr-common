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
        section_index → (3,) observed curvature [u1,u2,u3].  From FBG.
    wrenches : dict[int, np.ndarray]
        node_index → (6,) observed wrench [moment; force].  From F/T sensors.
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
    wrenches: Dict[int, np.ndarray] = field(default_factory=dict)

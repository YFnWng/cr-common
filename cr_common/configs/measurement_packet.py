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

    **Index convention** — every dict keyed by a rod node or section index
    (``positions``, ``poses``, ``strains``, ``node_strains``, ``wrenches``)
    uses **estimator-local** indices in ``[0, n_est_nodes)``.  For SOFA-driven
    pipelines the conversion from absolute rod indices happens at the bridge
    boundary (``sofa/bridge/packet.py``); hardware producers must follow the
    same convention.

    Attributes
    ----------
    timestamp : float
        Simulation or wall-clock time (s).
    dt : float
        Duration of this timestep (s).
    base_pose : np.ndarray, shape (7,), optional
        Base frame pose [x, y, z, qx, qy, qz, qw] at the estimator's proximal
        node (``rod.proximal_node_idx``, ``k_est = 0``).  Required when
        ``known_base_pose=True``; may be ``None`` otherwise.
    cable_disp : float, optional
        Cable displacement (mm).  Set when cable is displacement-controlled.
    cable_tensions : np.ndarray, shape (n_cables,), optional
        Per-cable tensions (N).  Set when cable is force-controlled.  Single-
        cable setups use a length-1 array.  ``None`` to let the estimator
        treat cable tension as a free variable.
    positions : dict[int, np.ndarray]
        estimator-local node index → (3,) observed position.  From MRI coils
        or EM position.
    poses : dict[int, np.ndarray]
        estimator-local node index → (7,) observed pose
        [x,y,z,qx,qy,qz,qw].  From EM coils.
    strains : dict[int, np.ndarray]
        estimator-local section index → (3,) observed midpoint curvature
        [u1,u2,u3].  From FBG.  Becomes a midpoint factor
        ``(S(k)+S(k+1))/2 = κ_obs[k]``.
    node_strains : dict[int, np.ndarray]
        estimator-local node index → (3,) observed curvature [u1,u2,u3].
        Becomes a ``PriorFactorVector(S(k), κ_obs)``.  Useful when a sensor
        directly measures curvature at a specific backbone node rather than
        a section midpoint.
    wrenches : dict[int, np.ndarray]
        estimator-local node index → (6,) observed wrench [moment; force].
        From F/T sensors.
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
    cable_tensions: Optional[np.ndarray] = None  # shape (n_cables,)
    positions: Dict[int, np.ndarray] = field(default_factory=dict)
    poses: Dict[int, np.ndarray] = field(default_factory=dict)
    strains: Dict[int, np.ndarray] = field(default_factory=dict)
    node_strains: Dict[int, np.ndarray] = field(default_factory=dict)
    wrenches: Dict[int, np.ndarray] = field(default_factory=dict)
    base_commands: Optional[np.ndarray] = None

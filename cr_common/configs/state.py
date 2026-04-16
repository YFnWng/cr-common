from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class NodeState:
    """Estimated state at a single node along the rod."""

    arclength: float                        # arc length from base (mm)
    pose: np.ndarray                        # (7,) [x,y,z, qx,qy,qz,qw] in world frame
    strain: np.ndarray                      # (3,) curvature/torsion [u1,u2,u3] (rad/mm)
    covariance: np.ndarray                  # (6,6) marginal pose covariance
    velocity: Optional[np.ndarray] = None        # (6,) body-centric twist ϖ — dynamic variants
    internal_force: Optional[np.ndarray] = None  # (3,) body-frame internal force N(k) — Kirchhoff force variant
    contact_force: Optional[np.ndarray] = None   # (3,) lumped contact force F(k) on section k

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3]

    @property
    def quaternion(self) -> np.ndarray:
        """[qx, qy, qz, qw]"""
        return self.pose[3:7]


@dataclass
class RodState:
    """Full estimated state of the Cosserat rod at one timestep."""

    timestamp: float
    nodes: List[NodeState]

    @property
    def tip(self) -> NodeState:
        return self.nodes[-1]

    @property
    def tip_position(self) -> np.ndarray:
        return self.tip.position

    @property
    def tip_position_std(self) -> np.ndarray:
        """1-sigma uncertainty on tip position (mm), derived from covariance diagonal."""
        return np.sqrt(np.diag(self.tip.covariance)[:3])

"""Abstract sensor protocol and shared data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class SofaGroundTruth:
    """Raw simulation state extracted from SOFA before any noise injection.

    Populated by ``sofa.bridge.reader.SofaReader`` and passed to sensors.
    """

    frame_poses: np.ndarray             # (n_frames, 7) Rigid3d [x,y,z,qx,qy,qz,qw]
    strain_coords: np.ndarray           # (n_sections, 3) estimation local-Z convention: [-κ_z, κ_y, τ_x]
    base_pose: np.ndarray               # (7,)
    cable_disp: float = 0.0             # mm (displacement-controlled mode)
    cable_tension: Optional[float] = None  # N (force-controlled mode; None = displacement mode)
    contact_force_body: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Per-section contact force in the rod body frame (N), shape (n_sections, 3).
    # Populated only when SofaReader is given constraint_solver and contact_listener.


@dataclass
class SensorReadings:
    """Noise-injected observations from one or more sensors for one timestep.

    Collected by a ``SensorSuite`` and assembled into a ``MeasurementPacket``.
    """

    positions: Dict[int, np.ndarray] = field(default_factory=dict)
    """node_index → (3,) observed position (mm).  From MRI coils or EM position-only."""

    poses: Dict[int, np.ndarray] = field(default_factory=dict)
    """node_index → (7,) observed pose [x,y,z,qx,qy,qz,qw].  From EM coils."""

    strains: Dict[int, np.ndarray] = field(default_factory=dict)
    """section_index → (3,) observed curvature [u1,u2,u3].  From FBG."""

    def merge(self, other: "SensorReadings") -> None:
        """Merge readings from *other* into this object (in-place, no overwrite)."""
        for k, v in other.positions.items():
            self.positions.setdefault(k, v)
        for k, v in other.poses.items():
            self.poses.setdefault(k, v)
        for k, v in other.strains.items():
            self.strains.setdefault(k, v)


@runtime_checkable
class AbstractSensor(Protocol):
    """Protocol that all sensor implementations must satisfy."""

    def observe(
        self,
        sofa_gt: SofaGroundTruth,
        t: float,
        dt: float,
    ) -> SensorReadings:
        """Sample noisy observations from *sofa_gt* at time *t*."""
        ...


class SensorSuite:
    """Aggregates multiple sensors and merges their readings."""

    def __init__(self, sensors: list) -> None:
        self._sensors = list(sensors)

    def observe(
        self,
        sofa_gt: SofaGroundTruth,
        t: float,
        dt: float,
    ) -> SensorReadings:
        merged = SensorReadings()
        for sensor in self._sensors:
            merged.merge(sensor.observe(sofa_gt, t, dt))
        return merged

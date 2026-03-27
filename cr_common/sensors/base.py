"""Abstract sensor protocol and shared data structures.

``SofaGroundTruth`` lives in ``state_estimation.sofa.bridge.packet``
to keep this module free of SOFA-specific types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, runtime_checkable

import numpy as np


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
        sofa_gt,
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
        sofa_gt,
        t: float,
        dt: float,
    ) -> SensorReadings:
        merged = SensorReadings()
        for sensor in self._sensors:
            merged.merge(sensor.observe(sofa_gt, t, dt))
        return merged

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

    def __init__(self, sensors: list, n_position_sensors: int = 0) -> None:
        self._sensors = list(sensors)
        self.n_position_sensors = n_position_sensors

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

    @classmethod
    def from_yaml(cls, cfg: dict, n_sofa_nodes: int) -> "SensorSuite":
        """Build a SensorSuite from the ``sensors`` section of a YAML config.

        Lazily imports simulated sensor classes so this module stays free of
        SOFA-specific imports at the module level.

        Parameters
        ----------
        cfg : dict
            Full parsed YAML config (expects ``cfg["sensors"]``).
        n_sofa_nodes : int
            Total number of SOFA rod nodes (= n_sections + 1).  Passed to
            ``CoilConfig`` for frame-to-node mapping.
        """
        from .simulated.coil_sensor import CoilConfig, EMCoilSensor, MRICoilSensor
        from .simulated.fbg_sensor import FBGConfig, FBGSensor

        sensor_cfg = cfg.get("sensors", {})
        em_cfg  = sensor_cfg.get("em_coils", {})
        mri_cfg = sensor_cfg.get("mri_coils", {})
        fbg_cfg = sensor_cfg.get("fbg", {})

        sensors = []

        em_indices = em_cfg.get("frame_indices", [])
        if em_indices:
            sensors.append(EMCoilSensor(
                CoilConfig(
                    frame_indices=em_indices,
                    position_std=em_cfg.get("position_std", 1e-4),
                    n_sofa_frames=n_sofa_nodes,
                    n_estimation_nodes=n_sofa_nodes,
                ),
                orientation_std=em_cfg.get("orientation_std", 1.0) * np.pi / 180.0,
            ))

        mri_indices = mri_cfg.get("frame_indices", [])
        if mri_indices:
            sensors.append(MRICoilSensor(
                CoilConfig(
                    frame_indices=mri_indices,
                    position_std=mri_cfg.get("position_std", 1e-4),
                    n_sofa_frames=n_sofa_nodes,
                    n_estimation_nodes=n_sofa_nodes,
                ),
            ))

        fbg_sections = fbg_cfg.get("section_indices", [])
        if fbg_sections:
            sensors.append(FBGSensor(
                FBGConfig(
                    section_indices=fbg_sections,
                    strain_std=fbg_cfg.get("strain_std", 10.0),
                ),
            ))

        n_position_sensors = len(em_indices) + len(mri_indices)
        return cls(sensors, n_position_sensors=n_position_sensors)

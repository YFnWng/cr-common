"""Unified robot control interface.

Single source of truth for joint definitions, angle encoding/decoding,
base/tendon command splitting, and sensor configuration.

No SOFA or ROS2 dependency — pure config + numpy.  Loaded from the
robot YAML (``catheter_ablation.yaml``).  Used by preprocessing, models,
controllers, and visualization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml


@dataclass
class JointDef:
    """Definition of one actuated joint."""
    name: str           # "insertion", "rotation", "cable_0"
    type: str           # "linear", "angular", "force", "displacement"
    lower: float        # lower limit (raw units)
    upper: float        # upper limit (raw units)
    rate: float         # max rate (m/s, deg/s, N/s)
    unit: str           # "m", "deg", "N"

    @property
    def is_angular(self) -> bool:
        return self.type == "angular"

    @property
    def is_tendon(self) -> bool:
        return self.type in ("force", "displacement")


class RobotInterface:
    """Single source of truth for robot control interface.

    Provides:
    - Joint definitions (names, types, limits, units)
    - Raw ↔ encoded conversions (deg → cos/sin for angular joints)
    - Base vs tendon command splitting
    - Base state encoding (raw → model format)
    - Sensor configuration access

    Parameters
    ----------
    joints : list of JointDef
    sensors : dict
        Raw sensor config (em_coils, mri_coils, fbg sections).
    insertion_direction : (3,) array
        Unit vector for insertion in local frame.
    """

    def __init__(self, joints: List[JointDef],
                 sensors: dict = None,
                 insertion_direction: np.ndarray = None):
        self.joints = list(joints)
        self._sensors = sensors or {}
        self.insertion_direction = (
            np.asarray(insertion_direction, dtype=float)
            if insertion_direction is not None
            else np.array([0.0, 0.0, 1.0]))

        # Precompute indices
        self._angular_idx = [i for i, j in enumerate(self.joints) if j.is_angular]
        self._tendon_idx = [i for i, j in enumerate(self.joints) if j.is_tendon]
        self._base_idx = [i for i, j in enumerate(self.joints) if not j.is_tendon]

        # Encoded dimensions: each angular joint expands 1 → 2
        self._encoded_base_dim = sum(
            2 if j.is_angular else 1 for j in self.joints if not j.is_tendon)
        self._tendon_dim = len(self._tendon_idx)
        self._encoded_cmd_dim = self._encoded_base_dim + self._tendon_dim

        # Base state: angular positions expand 1 → 2, velocities stay as-is
        n_base = len(self._base_idx)
        n_angular_base = sum(1 for i in self._base_idx if self.joints[i].is_angular)
        self._base_pos_dim = n_base + n_angular_base  # each angular adds 1 extra
        self._base_vel_dim = n_base

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RobotInterface":
        """Load from a robot YAML config file.

        Reads the ``actuation`` and ``sensors`` sections.
        """
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        act = cfg.get("actuation", {})
        cable_mode = act.get("cable_mode", "force")
        mode_cfg = act.get(cable_mode, {})

        joints = []

        # Insertion joint
        joints.append(JointDef(
            name="insertion",
            type="linear",
            lower=0.0,
            upper=float(act.get("max_travel", 0.08)),
            rate=float(act.get("insertion_speed", 0.03)),
            unit="m",
        ))

        # Rotation joint
        max_rot = float(act.get("max_rotation", 180.0))
        joints.append(JointDef(
            name="rotation",
            type="angular",
            lower=-max_rot,
            upper=max_rot,
            rate=float(act.get("rotation_speed", 30.0)),
            unit="deg",
        ))

        # Cable joints
        cable_locs = act.get("cable_locations", [])
        for i in range(len(cable_locs)):
            if cable_mode == "force":
                joints.append(JointDef(
                    name=f"cable_{i}",
                    type="force",
                    lower=float(mode_cfg.get("pull_min", 0.0)),
                    upper=float(mode_cfg.get("pull_max", 50.0)),
                    rate=float(mode_cfg.get("pull_increment", 3.0)),
                    unit="N",
                ))
            else:
                joints.append(JointDef(
                    name=f"cable_{i}",
                    type="displacement",
                    lower=float(mode_cfg.get("pull_min", 0.0)),
                    upper=float(mode_cfg.get("pull_max", 0.03)),
                    rate=float(mode_cfg.get("pull_increment", 0.003)),
                    unit="m",
                ))

        insertion_dir = act.get("insertion_direction", [0.0, 0.0, 1.0])
        sensors = cfg.get("sensors", {})

        return cls(joints, sensors=sensors,
                   insertion_direction=np.array(insertion_dir))

    # ------------------------------------------------------------------
    # Joint properties
    # ------------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def joint_lower(self) -> np.ndarray:
        """Raw joint lower limits."""
        return np.array([j.lower for j in self.joints], dtype=np.float64)

    @property
    def joint_upper(self) -> np.ndarray:
        """Raw joint upper limits."""
        return np.array([j.upper for j in self.joints], dtype=np.float64)

    @property
    def joint_rates(self) -> np.ndarray:
        return np.array([j.rate for j in self.joints], dtype=np.float64)

    @property
    def angular_indices(self) -> List[int]:
        """Indices of angular joints (in raw joint ordering)."""
        return list(self._angular_idx)

    @property
    def tendon_indices(self) -> List[int]:
        """Indices of tendon (cable) joints."""
        return list(self._tendon_idx)

    @property
    def base_indices(self) -> List[int]:
        """Indices of non-tendon (base actuator) joints."""
        return list(self._base_idx)

    @property
    def n_tendon(self) -> int:
        return self._tendon_dim

    # ------------------------------------------------------------------
    # Command encoding / decoding
    # ------------------------------------------------------------------

    def encode_command(self, raw_cmd: np.ndarray) -> np.ndarray:
        """Encode raw joint commands: angular joints (deg) → (cos, sin).

        Parameters
        ----------
        raw_cmd : (..., n_joints) raw commands in native units

        Returns
        -------
        encoded : (..., encoded_cmd_dim) with angular joints expanded
        """
        single = raw_cmd.ndim == 1
        if single:
            raw_cmd = raw_cmd.reshape(1, -1)

        parts = []
        for i, j in enumerate(self.joints):
            if j.is_angular:
                rad = np.deg2rad(raw_cmd[:, i:i + 1])
                parts.append(np.cos(rad))
                parts.append(np.sin(rad))
            else:
                parts.append(raw_cmd[:, i:i + 1])
        result = np.concatenate(parts, axis=-1)
        return result[0] if single else result

    def decode_command(self, encoded_cmd: np.ndarray) -> np.ndarray:
        """Decode encoded commands: (cos, sin) → angular joints (deg).

        Parameters
        ----------
        encoded_cmd : (..., encoded_cmd_dim)

        Returns
        -------
        raw_cmd : (..., n_joints) in native units
        """
        single = encoded_cmd.ndim == 1
        if single:
            encoded_cmd = encoded_cmd.reshape(1, -1)

        result = np.zeros((*encoded_cmd.shape[:-1], self.n_joints),
                          dtype=np.float64)
        enc_idx = 0
        for i, j in enumerate(self.joints):
            if j.is_angular:
                result[..., i] = np.degrees(
                    np.arctan2(encoded_cmd[..., enc_idx + 1],
                               encoded_cmd[..., enc_idx]))
                enc_idx += 2
            else:
                result[..., i] = encoded_cmd[..., enc_idx]
                enc_idx += 1
        return result[0] if single else result

    def split_command(self, encoded_cmd: np.ndarray):
        """Split encoded command into (base_cmd, tendon_cmd).

        Parameters
        ----------
        encoded_cmd : (encoded_cmd_dim,) or (N, encoded_cmd_dim)

        Returns
        -------
        base_cmd : (..., encoded_base_dim)
        tendon_cmd : (..., tendon_dim)
        """
        # Base commands come first (in encoded order), tendons last
        base = encoded_cmd[..., :self._encoded_base_dim]
        tendon = encoded_cmd[..., self._encoded_base_dim:]
        return base, tendon

    @property
    def encoded_base_dim(self) -> int:
        return self._encoded_base_dim

    @property
    def tendon_dim(self) -> int:
        return self._tendon_dim

    @property
    def encoded_cmd_dim(self) -> int:
        return self._encoded_cmd_dim

    # ------------------------------------------------------------------
    # Base state encoding / decoding
    # ------------------------------------------------------------------

    def encode_base_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Encode raw base state for model input.

        Parameters
        ----------
        raw_state : (..., 2*n_base) = [pos_0, pos_1, ..., vel_0, vel_1, ...]
            Positions in native units (m for linear, rad for angular),
            velocities in native rate units (m/s, rad/s).

        Returns
        -------
        encoded : (..., base_state_dim) with angular positions as (cos, sin)
        """
        n_base = len(self._base_idx)
        pos_raw = raw_state[..., :n_base]
        vel_raw = raw_state[..., n_base:2 * n_base]

        pos_parts = []
        for local_i, global_i in enumerate(self._base_idx):
            if self.joints[global_i].is_angular:
                # Angular position: value is in radians
                pos_parts.append(np.cos(pos_raw[..., local_i:local_i + 1]))
                pos_parts.append(np.sin(pos_raw[..., local_i:local_i + 1]))
            else:
                pos_parts.append(pos_raw[..., local_i:local_i + 1])

        pos_enc = np.concatenate(pos_parts, axis=-1)
        return np.concatenate([pos_enc, vel_raw], axis=-1)

    def decode_base_state(self, encoded_state: np.ndarray) -> np.ndarray:
        """Decode model base state to raw.

        Parameters
        ----------
        encoded_state : (..., base_state_dim)

        Returns
        -------
        raw_state : (..., 2*n_base) = [pos_0, ..., vel_0, ...]
        """
        n_base = len(self._base_idx)
        pos_enc = encoded_state[..., :self._base_pos_dim]
        vel = encoded_state[..., self._base_pos_dim:]

        pos_parts = []
        enc_idx = 0
        for local_i, global_i in enumerate(self._base_idx):
            if self.joints[global_i].is_angular:
                angle = np.arctan2(pos_enc[..., enc_idx + 1],
                                   pos_enc[..., enc_idx])
                pos_parts.append(angle[..., np.newaxis] if angle.ndim > 0
                                 else np.array([angle]))
                enc_idx += 2
            else:
                pos_parts.append(pos_enc[..., enc_idx:enc_idx + 1])
                enc_idx += 1

        pos_raw = np.concatenate(pos_parts, axis=-1)
        return np.concatenate([pos_raw, vel], axis=-1)

    @property
    def base_pos_dim(self) -> int:
        return self._base_pos_dim

    @property
    def base_vel_dim(self) -> int:
        return self._base_vel_dim

    @property
    def base_state_dim(self) -> int:
        return self._base_pos_dim + self._base_vel_dim

    # ------------------------------------------------------------------
    # Sensor access
    # ------------------------------------------------------------------

    @property
    def sensor_config(self) -> dict:
        return self._sensors

    def build_sensor_suite(self, n_sofa_nodes: int):
        """Build a SensorSuite from the sensor config."""
        from state_estimation.sensors.base import SensorSuite
        # SensorSuite.from_yaml expects a dict with top-level "sensors" key
        return SensorSuite.from_yaml({"sensors": self._sensors}, n_sofa_nodes)

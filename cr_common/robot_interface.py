"""Unified robot control interface.

Single source of truth for joint definitions, angle encoding/decoding,
base/tendon command splitting, sensor configuration, forward kinematics,
and base-state geometry (frame reconstruction + Jacobian).

No SOFA or ROS2 dependency — pure config + numpy.  Loaded from the
robot YAML (``catheter_ablation.yaml``).  Used by preprocessing, models,
controllers, and visualization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as _Rotation


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
                 insertion_direction: np.ndarray = None,
                 rod_length: float = None,
                 n_sections: int = None,
                 home_position: np.ndarray = None,
                 home_rotation: _Rotation = None,
                 encode_angles: bool = True):
        self.joints = list(joints)
        self._sensors = sensors or {}
        self._encode_angles = encode_angles
        self.insertion_direction = (
            np.asarray(insertion_direction, dtype=float)
            if insertion_direction is not None
            else np.array([0.0, 0.0, 1.0]))

        # Rod geometry (for FK and base frame reconstruction)
        self._rod_length = rod_length
        self._n_sections = n_sections
        self._ds = rod_length / n_sections if (rod_length and n_sections) else None
        self._home_position = (
            np.asarray(home_position, dtype=float)
            if home_position is not None
            else np.zeros(3))
        self._home_rotation = home_rotation or _Rotation.identity()

        # Precompute indices
        self._angular_idx = [i for i, j in enumerate(self.joints) if j.is_angular]
        self._tendon_idx = [i for i, j in enumerate(self.joints) if j.is_tendon]
        self._base_idx = [i for i, j in enumerate(self.joints) if not j.is_tendon]

        # Encoded dimensions: angular joints expand 1 → 2 only if encode_angles
        self._encoded_base_dim = sum(
            (2 if (j.is_angular and encode_angles) else 1)
            for j in self.joints if not j.is_tendon)
        self._tendon_dim = len(self._tendon_idx)
        self._encoded_cmd_dim = self._encoded_base_dim + self._tendon_dim

        # Base state: angular positions expand 1 → 2, velocities stay as-is
        n_base = len(self._base_idx)
        n_angular_base = sum(
            1 for i in self._base_idx
            if self.joints[i].is_angular and encode_angles)
        self._base_pos_dim = n_base + n_angular_base
        self._base_vel_dim = n_base

    @classmethod
    def from_yaml(cls, yaml_path: str,
                  encode_angles: bool = True) -> "RobotInterface":
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

        # Rod geometry
        rod = cfg.get("rod", {})
        rod_length = float(rod.get("length", 0.16))
        n_sections = int(rod.get("n_sections", 32))
        home_pos = np.array(rod.get("base_position", [0, 0, 0]), dtype=float)
        base_ori = np.array(
            rod.get("base_orientation_euler_xyz_deg", [0, 0, 0]), dtype=float)
        # Physical base orientation only — prefab rotation is SOFA-specific
        # and handled on the simulation side
        home_rot = _Rotation.from_euler("xyz", base_ori, degrees=True)

        return cls(joints, sensors=sensors,
                   insertion_direction=np.array(insertion_dir),
                   rod_length=rod_length, n_sections=n_sections,
                   home_position=home_pos, home_rotation=home_rot,
                   encode_angles=encode_angles)

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
        """Encode raw joint commands: angular joints (deg) → (cos, sin) or rad.

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
            if j.is_angular and self._encode_angles:
                rad = np.deg2rad(raw_cmd[:, i:i + 1])
                parts.append(np.cos(rad))
                parts.append(np.sin(rad))
            elif j.is_angular:
                parts.append(np.deg2rad(raw_cmd[:, i:i + 1]))
            else:
                parts.append(raw_cmd[:, i:i + 1])
        result = np.concatenate(parts, axis=-1)
        return result[0] if single else result

    def decode_command(self, encoded_cmd: np.ndarray) -> np.ndarray:
        """Decode encoded commands: (cos, sin) or rad → angular joints (deg).

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
            if j.is_angular and self._encode_angles:
                result[..., i] = np.degrees(
                    np.arctan2(encoded_cmd[..., enc_idx + 1],
                               encoded_cmd[..., enc_idx]))
                enc_idx += 2
            elif j.is_angular:
                result[..., i] = np.degrees(encoded_cmd[..., enc_idx])
                enc_idx += 1
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
                  or raw radians if encode_angles=False
        """
        n_base = len(self._base_idx)
        pos_raw = raw_state[..., :n_base]
        vel_raw = raw_state[..., n_base:2 * n_base]

        pos_parts = []
        for local_i, global_i in enumerate(self._base_idx):
            if self.joints[global_i].is_angular and self._encode_angles:
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
            if self.joints[global_i].is_angular and self._encode_angles:
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

    # ------------------------------------------------------------------
    # Rod geometry
    # ------------------------------------------------------------------

    @property
    def ds(self) -> float:
        """Section arc length (m)."""
        return self._ds

    @property
    def n_sections(self) -> int:
        return self._n_sections

    @property
    def n_base(self) -> int:
        """Number of raw base joints (before encoding)."""
        return len(self._base_idx)

    @property
    def base_state_raw_dim(self) -> int:
        """Raw base state dimension: n_base positions + n_base velocities."""
        return 2 * len(self._base_idx)

    @property
    def home_position(self) -> np.ndarray:
        return self._home_position.copy()

    @property
    def home_rotation(self) -> _Rotation:
        return self._home_rotation

    @property
    def home_pose_7(self) -> np.ndarray:
        """Home base pose as (7,) = [x, y, z, qx, qy, qz, qw]."""
        return np.concatenate([
            self._home_position, self._home_rotation.as_quat()])

    # ------------------------------------------------------------------
    # Base state std encoding
    # ------------------------------------------------------------------

    def encode_base_state_std(self, raw_pos_std: np.ndarray) -> np.ndarray:
        """Map raw base position std → encoded std.

        Angular positions get std=1.0 (cos/sin are bounded) when
        encode_angles=True, otherwise keep their raw std.

        Parameters
        ----------
        raw_pos_std : (n_base,) per-joint position std in native units

        Returns
        -------
        encoded_std : (base_pos_dim,)
        """
        parts = []
        for local_i, global_i in enumerate(self._base_idx):
            if self.joints[global_i].is_angular and self._encode_angles:
                parts.extend([1.0, 1.0])
            else:
                parts.append(float(raw_pos_std[local_i]))
        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    # TODO: clean up non-torch FK.
    def forward_kinematics(self, q, base_pose_7):
        """Compute all frame poses from strain via Cosserat exponential chain.

        Parameters
        ----------
        q : (n_sec, 3) curvature per section
        base_pose_7 : (7,) base frame [x,y,z,qx,qy,qz,qw]

        Returns
        -------
        T_all : list of gtsam.Pose3 — all frames (n_sec+1), from base to tip
        """
        import gtsam
        from state_estimation.utils import pose_from_vec7

        T = pose_from_vec7(base_pose_7)
        T_all = [T]
        ds = self._ds
        for k in range(len(q)):
            Omega_k = np.array([q[k, 0], q[k, 1], q[k, 2],
                                0.0, 0.0, 1.0]) * ds
            T = T.compose(gtsam.Pose3.Expmap(Omega_k))
            T_all.append(T)
        return T_all

    def forward_kinematics_with_jacobian(self, q, base_pose_7):
        """Compute all frame poses, per-frame body Jacobians, and adjoints.

        Uses the recursive body-frame Jacobian propagation:
            J_body[k+1] = Ad_inv(Exp(Ω_k)) @ (J_body[k] + T_r(Ω_k) @ Π · ds)

        Parameters
        ----------
        q : (n_sec, 3)
        base_pose_7 : (7,)

        Returns
        -------
        T_all : list of gtsam.Pose3 — all frames (n_sec+1)
        J_body_all : list of (6, n_sec*3) arrays — body Jacobian at each frame
        Ad_inv_all : list of (6, 6) arrays — Ad_inv(T_{base→frame_k}) per frame
        """
        import gtsam
        from state_estimation.utils import pose_from_vec7

        T_k = pose_from_vec7(base_pose_7)
        n_sec = len(q)
        ds = self._ds
        PI = np.vstack([np.eye(3), np.zeros((3, 3))])  # (6, 3)

        J_body = np.zeros((6, n_sec * 3), dtype=np.float64)
        T_base_to_k = gtsam.Pose3()  # identity

        T_all = [T_k]
        J_body_all = [J_body.copy()]
        Ad_inv_all = [np.eye(6, dtype=np.float64)]

        for k in range(n_sec):
            Omega_k = np.array([q[k, 0], q[k, 1], q[k, 2],
                                0.0, 0.0, 1.0]) * ds
            expOmega = gtsam.Pose3.Expmap(Omega_k)
            T_r = gtsam.Pose3.ExpmapDerivative(Omega_k)  # (6, 6)
            Ad_inv_exp = expOmega.inverse().AdjointMap()  # (6, 6)

            J_body[:, k * 3:(k + 1) * 3] += T_r @ PI * ds
            J_body = Ad_inv_exp @ J_body

            T_base_to_k = T_base_to_k.compose(expOmega)
            T_k = T_k.compose(expOmega)

            T_all.append(T_k)
            J_body_all.append(J_body.copy())
            Ad_inv_all.append(T_base_to_k.inverse().AdjointMap())

        return T_all, J_body_all, Ad_inv_all

    def forward_kinematics_batch(self, q_batch, base_frame_batch, X_ref=None):
        """Batch FK: strain → R9 tip pose (convenience for visualization).

        Parameters
        ----------
        q_batch : (N, n_sec, 3)
        base_frame_batch : (N, 7)
        X_ref : gtsam.Pose3 or None
            If None, tip poses are expressed in world frame.

        Returns
        -------
        xi_batch : (N, 9)
        """
        import gtsam
        from state_estimation.utils import SE3_to_R9

        if X_ref is None:
            X_ref = gtsam.Pose3.Identity()
        N = q_batch.shape[0]
        xi_batch = np.zeros((N, 9), dtype=np.float32)
        for i in range(N):
            T_all = self.forward_kinematics(q_batch[i], base_frame_batch[i])
            xi_batch[i] = SE3_to_R9(X_ref.between(T_all[-1]))
        return xi_batch

    # ------------------------------------------------------------------
    # Base frame geometry
    # ------------------------------------------------------------------

    def base_frame_from_encoded_state(self, encoded_base_pos: np.ndarray
                                      ) -> np.ndarray:
        """Reconstruct base frame (7,) from encoded base position.

        Parameters
        ----------
        encoded_base_pos : (base_pos_dim,) e.g. [ins, cos(rot), sin(rot)]

        Returns
        -------
        pose_7 : (7,) = [x, y, z, qx, qy, qz, qw]
        """
        ins = float(encoded_base_pos[0])
        rot_rad = float(np.arctan2(encoded_base_pos[2], encoded_base_pos[1]))

        world_dir = self._home_rotation.apply(self.insertion_direction)
        world_dir = world_dir / np.linalg.norm(world_dir)

        target_pos = self._home_position + world_dir * ins
        rotation = _Rotation.from_rotvec(rot_rad * world_dir)
        target_quat = (rotation * self._home_rotation).as_quat()
        return np.concatenate([target_pos, target_quat])

    def base_state_jacobian(self, encoded_base_pos: np.ndarray) -> np.ndarray:
        """Compute d(ξ_body)/d(encoded_base_pos): (6, base_pos_dim) matrix.

        Maps perturbations in encoded base position [ins, cos(rot), sin(rot)]
        or [ins, rot_rad] to body-frame twist ξ = [ω, v] (GTSAM convention).

        Right perturbation: T' = T * Exp(ξ).
        """
        d_body = self.insertion_direction.astype(np.float64)
        J = np.zeros((6, self._base_pos_dim), dtype=np.float64)
        J[3:6, 0] = d_body                  # d(v)/d(ins)
        if self._encode_angles:
            cos_rot = float(encoded_base_pos[1])
            sin_rot = float(encoded_base_pos[2])
            J[0:3, 1] = d_body * (-sin_rot)     # d(ω)/d(cos_rot)
            J[0:3, 2] = d_body * cos_rot        # d(ω)/d(sin_rot)
        else:
            J[0:3, 1] = d_body                   # d(ω)/d(rot_rad)
        return J

    # ------------------------------------------------------------------
    # Torch methods (GPU-compatible, batched)
    # ------------------------------------------------------------------

    def encode_command_torch(self, raw_cmd: 'torch.Tensor') -> 'torch.Tensor':
        """Encode raw commands on GPU: angular joints (deg) → (cos, sin) or rad.

        Parameters
        ----------
        raw_cmd : (K, n_joints) or (n_joints,) tensor

        Returns
        -------
        encoded : (K, encoded_cmd_dim) or (encoded_cmd_dim,) tensor
        """
        import torch
        single = raw_cmd.dim() == 1
        if single:
            raw_cmd = raw_cmd.unsqueeze(0)

        parts = []
        for i, j in enumerate(self.joints):
            if j.is_angular and self._encode_angles:
                rad = raw_cmd[:, i:i + 1] * (3.141592653589793 / 180.0)
                parts.append(torch.cos(rad))
                parts.append(torch.sin(rad))
            elif j.is_angular:
                parts.append(raw_cmd[:, i:i + 1] * (3.141592653589793 / 180.0))
            else:
                parts.append(raw_cmd[:, i:i + 1])
        result = torch.cat(parts, dim=-1)
        return result.squeeze(0) if single else result

    def forward_kinematics_torch(self, z_batch: 'torch.Tensor',
                                 base_pos_batch: 'torch.Tensor',
                                 encoder) -> 'torch.Tensor':
        """Batched differentiable FK: latent state → all frame transforms.

        Parameters
        ----------
        z_batch : (K, d) latent coordinates
        base_pos_batch : (K, base_pos_dim) encoded base position
        encoder : PCA encoder with ._components (d, n_strain) and ._mean

        Returns
        -------
        T_all : (K, n_sec+1, 4, 4) homogeneous transforms for all frames
        """
        import torch
        from state_estimation.utils import se3_exp_torch

        device = z_batch.device
        K = z_batch.shape[0]
        n_sec = self._n_sections

        # PCA decode: z → strain
        W = torch.tensor(encoder._components, dtype=torch.float32,
                         device=device)  # (d, n_strain)
        mean = torch.tensor(encoder._mean, dtype=torch.float32,
                            device=device)  # (n_strain,)
        q_flat = z_batch @ W + mean  # (K, n_strain)
        q = q_flat.reshape(K, n_sec, 3)  # (K, n_sec, 3)

        # Reconstruct base frame from encoded state
        T = self._base_frame_from_encoded_torch(base_pos_batch)  # (K, 4, 4)

        # Precompute ALL section exponentials in one batched call
        ds = self._ds
        omega_all = q * ds  # (K, n_sec, 3)
        v_all = torch.zeros(K, n_sec, 3, device=device, dtype=torch.float32)
        v_all[:, :, 2] = ds  # unit tangent along rod local Z axis
        xi_all = torch.cat([omega_all, v_all], dim=-1)  # (K, n_sec, 6)
        dT_all = se3_exp_torch(xi_all)  # (K, n_sec, 4, 4)

        # Sequential chain multiplication, storing all frames
        T_all = torch.empty(K, n_sec + 1, 4, 4, device=device,
                            dtype=torch.float32)
        T_all[:, 0] = T
        for k in range(n_sec):
            T = torch.matmul(T, dT_all[:, k])
            T_all[:, k + 1] = T

        return T_all

    def _base_frame_from_encoded_torch(self, encoded_base_pos: 'torch.Tensor'
                                       ) -> 'torch.Tensor':
        """Batched base frame reconstruction from encoded state.

        Parameters
        ----------
        encoded_base_pos : (K, base_pos_dim) e.g. (K, 3) = [ins, cos, sin]

        Returns
        -------
        T : (K, 4, 4) homogeneous transforms
        """
        import torch

        device = encoded_base_pos.device
        K = encoded_base_pos.shape[0]

        ins = encoded_base_pos[:, 0]          # (K,)
        if self._encode_angles:
            cos_rot = encoded_base_pos[:, 1]      # (K,)
            sin_rot = encoded_base_pos[:, 2]      # (K,)
            rot_rad = torch.atan2(sin_rot, cos_rot)
        else:
            rot_rad = encoded_base_pos[:, 1]      # (K,)
            cos_rot = torch.cos(rot_rad)
            sin_rot = torch.sin(rot_rad)

        # Home rotation as matrix
        home_R = torch.tensor(
            self._home_rotation.as_matrix(), dtype=torch.float32,
            device=device)  # (3, 3)
        home_pos = torch.tensor(
            self._home_position, dtype=torch.float32, device=device)  # (3,)
        ins_dir = torch.tensor(
            self.insertion_direction, dtype=torch.float32, device=device)  # (3,)

        # World insertion direction
        world_dir = home_R @ ins_dir  # (3,)
        world_dir = world_dir / world_dir.norm()

        # Position: home + ins * world_dir
        pos = home_pos.unsqueeze(0) + ins.unsqueeze(-1) * world_dir.unsqueeze(0)

        # Rotation: Rodrigues for axial rotation about world_dir
        # R_axial = I + sin(θ) [n]× + (1-cos(θ)) [n]×²
        from state_estimation.utils import skew_torch
        n_hat = skew_torch(world_dir.unsqueeze(0).expand(K, -1))  # (K, 3, 3)
        n_hat_sq = torch.matmul(n_hat, n_hat)
        I = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
        R_axial = (I + sin_rot[:, None, None] * n_hat
                   + (1 - cos_rot[:, None, None]) * n_hat_sq)
        # Full rotation: R_axial @ home_R
        R = torch.matmul(R_axial, home_R.unsqueeze(0).expand(K, -1, -1))

        # Assemble 4x4
        T = torch.zeros(K, 4, 4, device=device, dtype=torch.float32)
        T[:, :3, :3] = R
        T[:, :3, 3] = pos
        T[:, 3, 3] = 1.0
        return T

    @staticmethod
    def _rotation_matrix_to_quat_torch(R: 'torch.Tensor') -> 'torch.Tensor':
        """Convert (K, 3, 3) rotation matrices to (K, 4) quaternions [qx,qy,qz,qw]."""
        import torch
        # Shepperd's method (numerically stable)
        K = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        quat = torch.zeros(K, 4, device=R.device, dtype=R.dtype)

        # Case: trace > 0
        s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # 4*qw
        quat[:, 3] = 0.25 * s  # qw
        quat[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / s  # qx
        quat[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / s  # qy
        quat[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / s  # qz
        return quat

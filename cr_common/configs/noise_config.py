from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml

import numpy as np


@dataclass
class NoiseConfig:
    """Noise covariance parameters for the factor graph.

    All positional units in mm; angular units in rad.
    """

    # Process noise: integrated GP covariance per unit arc length.
    # Shape (6, 6) diagonal; matches GTSAM's [ω, v] Lie algebra convention:
    # first 3 = rotational DOF, last 3 = translational DOF.
    Qc: np.ndarray = field(
        default_factory=lambda: np.diag([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
    )

    # Base pose prior — very tight (known from hardware encoder / scene)
    base_pose_std: float = 1e-3     # applied to all 6 DOF equally

    # EM coil: full 6-DOF pose measurement
    em_position_std: float = 0.7    # mm
    em_orientation_std: float = 0.0175  # rad (~1 degree)

    # MRI coil: 3-DOF position only
    mri_position_std: float = 2.5   # mm

    # FBG strain: 3-DOF curvature/torsion (midpoint observation)
    fbg_strain_std: float = 0.01    # rad/mm

    # Tip strain prior: anchors S(n_sections) ≈ 0 (free-end boundary condition).
    # Replaces the old zero-mean section priors; only the tip is constrained.
    tip_strain_std: float = 0.05    # rad/mm

    # --- Dynamic kinematics (Teetaert 2025) ---
    # Body-centric velocity ϖ ∈ ℝ⁶ process noise (spatial GP, temporal GP).
    # Q1: 6×6 PSD for pose/velocity; Q3: 3×3 PSD for strain temporal evolution.
    velocity_std: float = 0.1       # ℝ⁶ body-centric twist (rad/s or mm/s)
    Q1: np.ndarray = field(
        default_factory=lambda: np.eye(6) * 1e-4   # 6×6 velocity PSD
    )
    Q3: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 1e-5   # 3×3 strain temporal PSD
    )

    # --- Quasi-static force (Ferguson 2026) ---
    # Internal wrench W ∈ ℝ⁶ mechanics soft constraint + cable tension.
    wrench_std: float = 1e-3        # soft mechanics constraint (N·mm / N)
    cable_tension_std: float = 0.5  # N — actuation uncertainty

    # ------------------------------------------------------------------ #
    @property
    def base_pose_cov(self) -> np.ndarray:
        s = self.base_pose_std
        return np.diag([s**2] * 6)

    @property
    def em_position_cov(self) -> np.ndarray:
        s = self.em_position_std
        return np.diag([s**2, s**2, s**2])

    @property
    def em_pose_cov(self) -> np.ndarray:
        sp = self.em_position_std
        sr = self.em_orientation_std
        # GTSAM PriorFactorPose3 error is Logmap([ω, v]): rotation first.
        return np.diag([sr**2, sr**2, sr**2, sp**2, sp**2, sp**2])

    @property
    def mri_position_cov(self) -> np.ndarray:
        s = self.mri_position_std
        return np.diag([s**2, s**2, s**2])

    @property
    def fbg_strain_cov(self) -> np.ndarray:
        s = self.fbg_strain_std
        return np.diag([s**2, s**2, s**2])

    @property
    def tip_strain_cov(self) -> np.ndarray:
        s = self.tip_strain_std
        return np.diag([s**2, s**2, s**2])

    @property
    def velocity_cov(self) -> np.ndarray:
        s = self.velocity_std
        return np.diag([s**2] * 6)

    @property
    def wrench_cov(self) -> np.ndarray:
        s = self.wrench_std
        return np.diag([s**2] * 6)

    @property
    def cable_tension_cov(self) -> np.ndarray:
        s = self.cable_tension_std
        return np.array([[s**2]])

    def make_Q9(self, ds: float) -> np.ndarray:
        """Build the 9×9 GP-derived covariance for one rod section (eq. 22, Lilge 2022).

        State ordering: [ω(0:3), v(3:6), κ(6:9)] — GTSAM Pose3 then curvature.
        Q_r (rotational) and Q_t (translational) are extracted from the diagonal
        of ``self.Qc``, which is already in [ω, v] order.
        """
        from ..factors.spatial_kinematics_factor import make_Q9
        Q_r = self.Qc[0:3, 0:3]
        Q_t = self.Qc[3:6, 3:6]
        return make_Q9(Q_r, Q_t, ds)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "NoiseConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        noise = cfg.get("noise", {})
        sensors = cfg.get("sensors", {})

        Qc_r = float(noise.get("Qc_rotation", 1e-6))
        Qc_t = float(noise.get("Qc_translation", 1e-4))
        Qc = np.diag([Qc_r, Qc_r, Qc_r, Qc_t, Qc_t, Qc_t])  # [ω, v] order

        em = sensors.get("em_coils", {})
        mri = sensors.get("mri_coils", {})
        fbg = sensors.get("fbg", {})

        return cls(
            Qc=Qc,
            base_pose_std=float(noise.get("base_pose_std", 1e-3)),
            em_position_std=float(em.get("position_std", 0.7)),
            em_orientation_std=float(em.get("orientation_std", 1.0)) * np.pi / 180.0,
            mri_position_std=float(mri.get("position_std", 2.5)),
            fbg_strain_std=float(fbg.get("strain_std", 0.01)),
            tip_strain_std=float(noise.get("tip_strain_std", 0.05)),
            velocity_std=float(noise.get("velocity_std", 0.1)),
            wrench_std=float(noise.get("wrench_std", 1e-3)),
            cable_tension_std=float(noise.get("cable_tension_std", 0.5)),
        )

    @classmethod
    def simulation_defaults(cls) -> "NoiseConfig":
        """Conservative defaults for SOFA simulation experiments."""
        return cls()

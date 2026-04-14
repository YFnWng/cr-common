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
        default_factory=lambda: np.eye(6) * 1e-4   # 6×6 acceleration PSD
    )
    Q3: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 1e-5   # 3×3 strain temporal PSD
    )

    # --- Quasi-static force (Ferguson 2026) ---
    # Internal wrench W ∈ ℝ⁶ mechanics soft constraint + cable tension.
    Q4: np.ndarray = field(
        default_factory=lambda: np.eye(6) * 1e-4   # 6×6 cosserat PSD
    )
    Q5: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 1e-3   # 3×3 external wrench spatial PSD
    )
    wrench_std: float = 1e-3        # legacy 6-DOF soft mechanics constraint
    moment_equilibrium_std: float = 1e-2   # N·mm — 3-DOF moment ODE residual σ (discretization noise)
    force_equilibrium_std: float = 1e-2    # N   — 3-DOF force ODE residual σ
    # Tip boundary condition: exact physical constraint (no discretization error).
    # Much tighter than ODE noise; anchors the tip and lets the Cosserat moment
    # chain determine all 33 strain nodes without a free zigzag parameter.
    # bc_moment_std << moment_equilibrium_std to suppress the alternating null-space.
    bc_moment_std: float = 0.1             # N·mm — tip BC moment residual σ
    bc_force_std: float = 0.1             # N   — tip BC force residual σ
    contact_force_std: float = 10.0        # N  — weak prior on lumped contact force F(k)
    tip_contact_force_std: float = 10.0   # N  — looser prior for tip node (allows concentrated contact)
    cable_tension_std: float = 0.5         # N  — actuation uncertainty per tendon
    temporal_strain_std: float = 0.0       # rad/m — temporal smoothing; 0 = disabled

    # --- Proximal-boundary identification (Wang manuscript) ---
    theta_prior_std: float = 10.0          # loose zero-mean prior on θ
    temporal_pose_std: float = 0.01        # m/rad — weak continuity on X across time steps
    temporal_force_std: float = 0.0        # N — temporal smoothing on N/F; 0 = disabled

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
    def moment_equilibrium_cov(self) -> np.ndarray:
        s = self.moment_equilibrium_std
        return np.diag([s**2] * 3)

    @property
    def force_equilibrium_cov(self) -> np.ndarray:
        s = self.force_equilibrium_std
        return np.diag([s**2] * 3)

    @property
    def bc_moment_cov(self) -> np.ndarray:
        s = self.bc_moment_std
        return np.diag([s**2] * 3)

    @property
    def bc_force_cov(self) -> np.ndarray:
        s = self.bc_force_std
        return np.diag([s**2] * 3)

    @property
    def contact_force_cov(self) -> np.ndarray:
        s = self.contact_force_std
        return np.diag([s**2] * 3)

    @property
    def tip_contact_force_cov(self) -> np.ndarray:
        s = self.tip_contact_force_std
        return np.diag([s**2] * 3)

    def cable_tensions_cov(self, n_tendons: int) -> np.ndarray:
        """Diagonal covariance for a vector of n_tendons cable tensions."""
        s = self.cable_tension_std
        return np.diag([s**2] * n_tendons)

    def theta_prior_cov(self, theta_dim: int) -> np.ndarray:
        """Diagonal covariance for the θ prior (proximal-boundary parameters)."""
        s = self.theta_prior_std
        return np.diag([s**2] * theta_dim)

    @property
    def temporal_pose_cov(self) -> np.ndarray:
        """6×6 diagonal covariance for the temporal pose continuity prior."""
        s = self.temporal_pose_std
        return np.diag([s**2] * 6)

    @property
    def cable_tension_cov(self) -> np.ndarray:
        s = self.cable_tension_std
        return np.array([[s**2]])

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

        # cable_tension_std: actuation.force sub-dict takes precedence over noise section
        act = cfg.get("actuation", {})
        cable_mode = act.get("cable_mode", "displacement")
        mode_cfg = act.get(cable_mode, {})
        cable_tension_std = float(
            mode_cfg.get("cable_tension_std",
                         noise.get("cable_tension_std", 0.5))
        )

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
            moment_equilibrium_std=float(noise.get("moment_equilibrium_std", 1e-2)),
            force_equilibrium_std=float(noise.get("force_equilibrium_std", 1e-2)),
            bc_moment_std=float(noise.get("bc_moment_std", 0.1)),
            bc_force_std=float(noise.get("bc_force_std", 0.1)),
            contact_force_std=float(noise.get("contact_force_std", 10.0)),
            tip_contact_force_std=float(noise.get("tip_contact_force_std",
                                                    noise.get("contact_force_std", 10.0))),
            cable_tension_std=cable_tension_std,
            temporal_strain_std=float(noise.get("temporal_strain_std", 0.0)),
            theta_prior_std=float(noise.get("theta_prior_std", 10.0)),
            temporal_pose_std=float(noise.get("temporal_pose_std", 0.01)),
            temporal_force_std=float(noise.get("temporal_force_std", 0.0)),
        )

    @classmethod
    def simulation_defaults(cls) -> "NoiseConfig":
        """Conservative defaults for SOFA simulation experiments."""
        return cls()

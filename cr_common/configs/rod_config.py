from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class RodConfig:
    """Geometric and mechanical parameters of a single Cosserat rod.

    All lengths in mm, forces in N, angles in rad.
    """

    length: float               # total arc length (mm)
    n_sections: int = 32        # number of sections (strain variables)
    young_modulus: float = 8e5  # Pa
    poisson_ratio: float = 0.38
    beam_radius: float = 1.45   # mm
    kirchhoff: bool = True      # True → lock shear/extension DOF (3-DOF strain)

    # Base pose as a 4×4 homogeneous matrix; default = identity (origin, aligned with Z)
    base_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )

    # Tendon routing geometry (Rucker & Webster 2011).
    # All coordinates in the body frame, units mm.
    # For straight tendons (SOFA default): d_offsets = dd_offsets = zeros.
    # For curved routing: supply the derivatives dr_i/ds and d²r_i/ds² evaluated
    # at each section — the factor code uses the *constant* values stored here,
    # so update them per-node if the routing is arc-length-dependent.
    tendon_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )  # (n_tendons, 2)  [x_i, y_i] offset in cross-section (mm)
    tendon_d_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )  # (n_tendons, 2)  dr_i/ds  — zero for straight routing
    tendon_dd_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )  # (n_tendons, 2)  d²r_i/ds² — zero for straight routing

    @property
    def n_tendons(self) -> int:
        """Number of tendons."""
        return int(self.tendon_offsets.shape[0])

    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        """Number of pose nodes = n_sections + 1 (one per section endpoint)."""
        return self.n_sections + 1

    @property
    def section_length(self) -> float:
        """Arc length of each uniform section (mm)."""
        return self.length / self.n_sections

    @property
    def strain_dim(self) -> int:
        """Dimension of the strain variable per section (3 for Kirchhoff, 6 otherwise)."""
        return 3 if self.kirchhoff else 6

    # ------------------------------------------------------------------ #
    # Cross-section geometry and stiffness
    # ------------------------------------------------------------------ #

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area A = π r²  (mm²)."""
        return np.pi * self.beam_radius ** 2

    @property
    def second_moment_of_area(self) -> float:
        """Second moment of area I = π r⁴ / 4  (mm⁴)."""
        return np.pi * self.beam_radius ** 4 / 4.0

    @property
    def polar_moment_of_area(self) -> float:
        """Polar second moment of area J = π r⁴ / 2 = 2I  (mm⁴)."""
        return np.pi * self.beam_radius ** 4 / 2.0

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2(1 + ν))."""
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def shear_correction_factor(self) -> float:
        """Timoshenko shear correction factor κ_s for a solid circular cross-section.

        Uses Cowper's (1966) formula: κ_s = 6(1+ν) / (7+6ν).
        At ν=0 → 6/7 ≈ 0.857; at ν=0.5 → 0.900.
        """
        nu = self.poisson_ratio
        return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Cross-section constitutive stiffness matrix K.

        Relates generalised strain ε to internal wrench σ = K ε.
        Ordered to match the GTSAM/Lilge [ω, v] Lie algebra convention
        (rotational DOF first, translational DOF last):

        Kirchhoff (3×3, ε = [κ₁, κ₂, τ]):
            K = diag([EI, EI, GJ])

        Full Cosserat (6×6, ε = [κ₁, κ₂, τ, γ₁, γ₂, ε_z]):
            K = diag([EI, EI, GJ, κ_s·GA, κ_s·GA, EA])

        Units: bending/torsion stiffness in N·mm²  (with lengths in mm, E in N/mm²).
               shear/axial stiffness in N.
        """
        E  = self.young_modulus
        G  = self.shear_modulus
        I  = self.second_moment_of_area
        J  = self.polar_moment_of_area
        A  = self.cross_section_area
        ks = self.shear_correction_factor

        EI = E * I
        GJ = G * J
        if self.kirchhoff:
            return np.diag([EI, EI, GJ])
        else:
            GA_s = ks * G * A
            EA   = E * A
            return np.diag([EI, EI, GJ, GA_s, GA_s, EA])

    @property
    def stiffness_matrix_inv(self) -> np.ndarray:
        """Inverse of the cross-section stiffness matrix K⁻¹ (compliance matrix)."""
        return np.linalg.inv(self.stiffness_matrix)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "RodConfig":
        """Construct from the ``rod`` section of a YAML config file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        rod = cfg.get("rod", {})
        return cls(
            length=float(rod.get("length", 160.0)),
            n_sections=int(rod.get("n_sections", 32)),
            young_modulus=float(rod.get("young_modulus", 8e5)),
            poisson_ratio=float(rod.get("poisson_ratio", 0.38)),
            beam_radius=float(rod.get("beam_radius", 1.45)),
            kirchhoff=bool(rod.get("kirchhoff", True)),
        )

    @classmethod
    def from_sofa_params(cls, params) -> "RodConfig":
        """Construct from a SOFA ``Parameters`` composite object."""
        g = params.beam_geo_params
        p = params.beam_physics_params
        return cls(
            length=float(g.beam_length),
            n_sections=int(g.nb_section),
            young_modulus=float(p.young_modulus),
            poisson_ratio=float(p.poisson_ratio),
            beam_radius=float(p.beam_radius),
        )

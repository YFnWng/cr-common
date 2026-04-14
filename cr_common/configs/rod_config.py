from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class RodConfig:
    """Geometric and mechanical parameters of a single Cosserat rod.

    All values in SI units: metres, Pa, N, rad/m.
    YAML config stores values in SI directly.
    """

    length: float               # total arc length (m)
    n_sections: int = 32        # number of sections (strain variables)
    young_modulus: float = 8e5  # Pa
    poisson_ratio: float = 0.38
    beam_radius: float = 0.00145  # m
    kirchhoff: bool = True      # True → lock shear/extension DOF (3-DOF strain)

    # Base pose as a 4×4 homogeneous matrix; default = identity (origin, aligned with Z)
    base_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )

    # Tendon routing geometry (Rucker & Webster 2011).
    # All coordinates in the body frame, units m.
    # For straight tendons (SOFA default): d_offsets = dd_offsets = zeros.
    tendon_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )  # (n_tendons, 2)  [x_i, y_i] offset in cross-section (m)
    tendon_d_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )  # (n_tendons, 2)  dr_i/ds  — zero for straight routing
    tendon_dd_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )  # (n_tendons, 2)  d²r_i/ds² — zero for straight routing

    # ------------------------------------------------------------------ #
    # Proximal boundary identification (Wang manuscript)
    # ------------------------------------------------------------------ #

    # Actuator reference pose (SE(3)) used to compute ξ = Log(X_ref⁻¹·X(0))
    # in the basis dictionary.  Defaults to identity → ξ is the body-frame
    # log of X(0) relative to the world origin.
    X_ref: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )  # (4, 4) homogeneous matrix

    # Dimension of u_ext (base-actuator external commands, e.g. insertion
    # translation, base rotation).  Concatenated with Q to form u_act.
    n_base_commands: int = 0

    # SOFA node index at which the estimator's proximal boundary lives.
    # 0 → full rod (matches existing qs_force behaviour).  >0 → partial
    # rod; estimator covers SOFA nodes [proximal_node_idx .. n_nodes-1].
    proximal_node_idx: int = 0

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
        """Arc length of each uniform section (m)."""
        return self.length / self.n_sections

    @property
    def strain_dim(self) -> int:
        """Dimension of the strain variable per section (3 for Kirchhoff, 6 otherwise)."""
        return 3 if self.kirchhoff else 6

    @property
    def n_est_nodes(self) -> int:
        """Number of nodes in the estimator's partial-rod graph.

        Equals ``n_nodes - proximal_node_idx``.  When ``proximal_node_idx=0``
        this equals ``n_nodes`` (full rod).
        """
        return self.n_nodes - int(self.proximal_node_idx)

    def theta_dim(self, terms_enabled: "dict[str, bool] | None" = None) -> int:
        """Dimension of the proximal-boundary parameter vector θ.

        ``terms_enabled`` is a dict of the basis-dictionary toggles from the
        YAML (``bias``, ``linear_xi``, ``linear_s``, ``linear_u``, ``quad_xi_xi``,
        ``cross_xi_s``, ``cross_xi_u``, ``quad_s_s``, ``cross_s_u``,
        ``quad_u_u``).  If ``None``, assumes all terms enabled.

        Let m = ``n_tendons + n_base_commands`` (control-input dimension).
        Each enabled term contributes to θ as follows:

            bias         : 6
            linear_xi    : 36
            linear_s     : 18
            linear_u     : 6m
            quad_xi_xi   : 126   (6 × vech(ξξᵀ) with dim(vech) = 21)
            cross_xi_s   : 108   (6 × 18)
            cross_xi_u   : 36m
            quad_s_s     : 36    (6 × vech(ssᵀ) with dim(vech) = 6)
            cross_s_u    : 18m
            quad_u_u     : 3m(m+1)   (6 × m(m+1)/2)
        """
        m = int(self.n_tendons) + int(self.n_base_commands)
        if terms_enabled is None:
            terms_enabled = {k: True for k in (
                "bias", "linear_xi", "linear_s", "linear_u",
                "quad_xi_xi", "cross_xi_s", "cross_xi_u",
                "quad_s_s", "cross_s_u", "quad_u_u",
            )}
        dims = {
            "bias":       6,
            "linear_xi":  36,
            "linear_s":   18,
            "linear_u":   6 * m,
            "quad_xi_xi": 126,
            "cross_xi_s": 108,
            "cross_xi_u": 36 * m,
            "quad_s_s":   36,
            "cross_s_u":  18 * m,
            "quad_u_u":   3 * m * (m + 1),
        }
        return sum(d for k, d in dims.items() if terms_enabled.get(k, False))

    # ------------------------------------------------------------------ #
    # Cross-section geometry and stiffness (all SI)
    # ------------------------------------------------------------------ #

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area A = π r²  (m²)."""
        return np.pi * self.beam_radius ** 2

    @property
    def second_moment_of_area(self) -> float:
        """Second moment of area I = π r⁴ / 4  (m⁴)."""
        return np.pi * self.beam_radius ** 4 / 4.0

    @property
    def polar_moment_of_area(self) -> float:
        """Polar second moment of area J = π r⁴ / 2 = 2I  (m⁴)."""
        return np.pi * self.beam_radius ** 4 / 2.0

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2(1 + ν))  (Pa)."""
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
        """Cross-section constitutive stiffness matrix K  (SI: N·m²).

        Relates generalised strain ε to internal wrench σ = K ε.

        Kirchhoff (3×3, ε = [κ₁, κ₂, τ]):
            K = diag([EI, EI, GJ])       units: N·m²

        Full Cosserat (6×6, ε = [κ₁, κ₂, τ, γ₁, γ₂, ε_z]):
            K = diag([EI, EI, GJ, κ_s·GA, κ_s·GA, EA])
        """
        E  = self.young_modulus          # Pa = N/m²
        G  = self.shear_modulus          # Pa
        I  = self.second_moment_of_area  # m⁴
        J  = self.polar_moment_of_area   # m⁴
        A  = self.cross_section_area     # m²
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

        X_ref_raw = rod.get("X_ref", None)
        if X_ref_raw is not None:
            X_ref = np.asarray(X_ref_raw, dtype=float)
            if X_ref.shape != (4, 4):
                raise ValueError(f"rod.X_ref must be 4x4, got {X_ref.shape}")
        else:
            X_ref = np.eye(4, dtype=float)

        return cls(
            length=float(rod.get("length", 0.16)),
            n_sections=int(rod.get("n_sections", 32)),
            young_modulus=float(rod.get("young_modulus", 8e5)),
            poisson_ratio=float(rod.get("poisson_ratio", 0.38)),
            beam_radius=float(rod.get("beam_radius", 0.00145)),
            kirchhoff=bool(rod.get("kirchhoff", True)),
            X_ref=X_ref,
            n_base_commands=int(rod.get("n_base_commands", 0)),
            proximal_node_idx=int(rod.get("proximal_node_idx", 0)),
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

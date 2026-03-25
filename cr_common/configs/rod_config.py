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

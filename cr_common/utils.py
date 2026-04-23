"""
Utility functions for SE(3), including the continuous 6D rotation
representation from Zhou et al. 2020.
"""
from __future__ import annotations

import numpy as np
import gtsam

# Identity R9 representation: [R[:,0]; R[:,1]; t] at identity pose
R9_IDENTITY = np.array([1., 0., 0., 0., 1., 0., 0., 0., 0.])

def pose_from_vec7(v: np.ndarray) -> gtsam.Pose3:
    """Convert [x, y, z, qx, qy, qz, qw] → gtsam.Pose3."""
    qx, qy, qz, qw = float(v[3]), float(v[4]), float(v[5]), float(v[6])
    rot = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
    t   = np.array([float(v[0]), float(v[1]), float(v[2])])
    return gtsam.Pose3(rot, t)


def rot3_to_quat(rot: gtsam.Rot3) -> np.ndarray:
    """Return [qx, qy, qz, qw] from gtsam.Rot3."""
    q = rot.toQuaternion()
    return np.array([q.x(), q.y(), q.z(), q.w()])

def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix: skew(v) a = v × a."""
    return np.array([[ 0.,    -v[2],  v[1]],
                     [ v[2],   0.,   -v[0]],
                     [-v[1],   v[0],  0.  ]])


# ---------------------------------------------------------------------------
# Continuous 6D rotation representation (Zhou et al. 2020)
# ---------------------------------------------------------------------------

def SE3_to_R9(X: gtsam.Pose3) -> np.ndarray:
    """Convert SE(3) pose to continuous 9D representation.

    Returns [R[:,0]; R[:,1]; t] ∈ R⁹ where R is the rotation matrix
    and t is the translation vector.  This representation is continuous
    everywhere on SO(3), unlike Logmap which is discontinuous at π.
    """
    R = X.rotation().matrix()
    t = X.translation()
    return np.concatenate([R[:, 0], R[:, 1], t])


def SE3_to_R9_centered(X: gtsam.Pose3) -> np.ndarray:
    """Like SE3_to_R9 but centered so identity maps to zero.

    Subtracts [1,0,0, 0,1,0, 0,0,0] so that Pose3() → [0,...,0].
    Use with zero_at_zero=True gated architectures.
    """
    return SE3_to_R9(X) - R9_IDENTITY


def R9_jacobian(X: gtsam.Pose3) -> np.ndarray:
    """Jacobian of R9 representation w.r.t. right perturbation.

    Returns ∂r/∂δ ∈ R^{9×6} where δ = [ω; v] is the 6D tangent-space
    perturbation: X → X · Exp(δ).

    For a right perturbation R → R·(I + [ω]×):
        ∂(R·e_k)/∂ω = R · [ω]× · e_k  evaluated at ω=0
        → column j is R · (e_j × e_k)

    For translation: t → t + R·v, so ∂t/∂v = R.
    """
    R = X.rotation().matrix()
    J = np.zeros((9, 6))

    # R·Exp(ω) ≈ R·(I + [ω]×), so d(R·e_k)/dω = R · [ω]× · e_k
    # [ω]× · e_k = ω × e_k.  Column j of the Jacobian: R · (e_j × e_k)
    #
    # ∂R[:,0]/∂ω: e_j × e_0 for j=0,1,2:
    #   e_0×e_0=0, e_1×e_0=-e_2, e_2×e_0=e_1  → [[0,0,0],[0,0,1],[0,-1,0]]
    J[0:3, 0:3] = R @ np.array([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., -1., 0.]])

    # ∂R[:,1]/∂ω: e_j × e_1 for j=0,1,2:
    #   e_0×e_1=e_2, e_1×e_1=0, e_2×e_1=-e_0  → [[0,0,-1],[0,0,0],[1,0,0]]
    J[3:6, 0:3] = R @ np.array([[0., 0., -1.],
                                 [0., 0., 0.],
                                 [1., 0., 0.]])

    # ∂t/∂v = R
    J[6:9, 3:6] = R

    return J
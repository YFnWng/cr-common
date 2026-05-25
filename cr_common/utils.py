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


def dR9_dxi_right(T: gtsam.Pose3) -> np.ndarray:
    """Jacobian of SE3_to_R9 w.r.t. right perturbation T → T·Exp(δξ).

    Returns (9, 6) matrix where ξ = [ω; v] ∈ se(3).

    R9 = [R·e₀; R·e₁; t].  Under T → T·Exp(δξ):
        R → R·exp([ω]×), t → t + R·v
    """
    R = T.rotation().matrix()
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    dR9 = np.zeros((9, 6))
    dR9[:3, :3] = -R @ skew(e0)
    dR9[3:6, :3] = -R @ skew(e1)
    dR9[6:9, 3:6] = R
    return dR9


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


def R9_to_rotation(r9_row: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: R9 [R[:,0]; R[:,1]; t] → 3×3 rotation matrix."""
    e0 = r9_row[:3].copy()
    e0 /= np.linalg.norm(e0) + 1e-12
    e1 = r9_row[3:6] - np.dot(r9_row[3:6], e0) * e0
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(e0, e1)
    return np.column_stack([e0, e1, e2])


def r9_logmap_delta(r9_prev: np.ndarray, r9_curr: np.ndarray) -> np.ndarray:
    """Compute se(3) Logmap delta between consecutive R9 poses.

    ``Logmap(Pose(r9_prev)^{-1} * Pose(r9_curr))`` for each row.

    Parameters
    ----------
    r9_prev, r9_curr : (N, 9) arrays — R9 representation [R[:,0]; R[:,1]; t]

    Returns
    -------
    (N, 6) array — se(3) tangent vectors [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    """
    N = len(r9_prev)
    delta = np.zeros((N, 6))
    for i in range(N):
        R_p = gtsam.Rot3(R9_to_rotation(r9_prev[i]))
        t_p = gtsam.Point3(r9_prev[i, 6:9])
        R_c = gtsam.Rot3(R9_to_rotation(r9_curr[i]))
        t_c = gtsam.Point3(r9_curr[i, 6:9])
        P_prev = gtsam.Pose3(R_p, t_p)
        P_curr = gtsam.Pose3(R_c, t_c)
        delta[i] = gtsam.Pose3.Logmap(P_prev.between(P_curr))
    return delta


def R9_integrate_step(xi_prev_r9: np.ndarray, delta_se3: np.ndarray) -> np.ndarray:
    """Integrate one step: X_curr = X_prev * Exp(delta).

    Parameters
    ----------
    xi_prev_r9 : (9,) R9 representation [R[:,0]; R[:,1]; t]
    delta_se3 : (6,) se(3) tangent vector [omega; v]

    Returns
    -------
    xi_curr_r9 : (9,) R9 representation of X_curr
    """
    R_prev = R9_to_rotation(xi_prev_r9)
    t_prev = xi_prev_r9[6:9]
    X_prev = gtsam.Pose3(gtsam.Rot3(R_prev), gtsam.Point3(t_prev))
    X_curr = X_prev.compose(gtsam.Pose3.Expmap(delta_se3))
    return SE3_to_R9(X_curr)


def world_velocity_to_body(pose_quat: np.ndarray,
                           vel_world: np.ndarray) -> np.ndarray:
    """Convert world-frame Rigid3d velocity to body-frame se(3) velocity.

    Parameters
    ----------
    pose_quat : (7,) pose [x, y, z, qx, qy, qz, qw]
    vel_world : (6,) world-frame velocity [vx, vy, vz, wx, wy, wz]

    Returns
    -------
    eta : (6,) body-frame se(3) velocity [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    """
    R = gtsam.Rot3.Quaternion(
        float(pose_quat[6]), float(pose_quat[3]),
        float(pose_quat[4]), float(pose_quat[5]),
    ).matrix()
    v_body = R.T @ vel_world[:3]
    omega_body = R.T @ vel_world[3:6]
    return np.concatenate([omega_body, v_body])
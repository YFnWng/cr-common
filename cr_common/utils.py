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


def pose3_to_vec7(X: gtsam.Pose3) -> np.ndarray:
    """Convert gtsam.Pose3 → [x, y, z, qx, qy, qz, qw]."""
    t = X.translation()
    q = rot3_to_quat(X.rotation())
    return np.concatenate([t, q])

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


# ---------------------------------------------------------------------------
# Torch SE(3) utilities — batched, differentiable, GPU-compatible
# Convention: xi = [omega(3), v(3)] matching GTSAM (rotation first)
# Ported from github.com/YFnWng/continuumSim/SE3.py with singularity handling
# ---------------------------------------------------------------------------

import torch


def skew_torch(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix.

    Parameters
    ----------
    v : (..., 3)

    Returns
    -------
    S : (..., 3, 3)
    """
    S = torch.zeros(*v.shape, 3, device=v.device, dtype=v.dtype)
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] = v[..., 1]
    S[..., 1, 2] = -v[..., 0]
    S[..., 1, 0] = v[..., 2]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] = v[..., 0]
    return S


def se3_exp_torch(xi: torch.Tensor) -> torch.Tensor:
    """Batched exponential map se(3) → SE(3).

    Uses Rodrigues' formula with robust handling at theta → 0.

    Parameters
    ----------
    xi : (..., 6) = [omega(3), v(3)]
        Lie algebra element. omega = rotation, v = translation.

    Returns
    -------
    T : (..., 4, 4) homogeneous transformation matrix
    """
    omega = xi[..., :3]  # (..., 3) rotation
    v = xi[..., 3:]      # (..., 3) translation

    theta = omega.norm(dim=-1, keepdim=True)  # (..., 1)
    theta_sq = theta ** 2
    theta_cu = theta_sq * theta

    # Robust coefficients with Taylor expansion at theta ≈ 0
    # sin(θ)/θ → 1 - θ²/6
    # (1 - cos(θ))/θ² → 1/2 - θ²/24
    # (θ - sin(θ))/θ³ → 1/6 - θ²/120
    small = (theta < 1e-7).squeeze(-1)

    safe_theta = torch.where(theta < 1e-7, torch.ones_like(theta), theta)
    safe_theta_sq = safe_theta ** 2
    safe_theta_cu = safe_theta_sq * safe_theta

    alpha = torch.sin(theta) / safe_theta
    beta = (1.0 - torch.cos(theta)) / safe_theta_sq
    gamma = (theta - torch.sin(theta)) / safe_theta_cu

    # Taylor limits for small theta
    alpha = torch.where(theta < 1e-7, 1.0 - theta_sq / 6.0, alpha)
    beta = torch.where(theta < 1e-7, 0.5 - theta_sq / 24.0, beta)
    gamma = torch.where(theta < 1e-7, 1.0 / 6.0 - theta_sq / 120.0, gamma)

    # Skew-symmetric matrix of omega: (..., 3, 3)
    omega_hat = skew_torch(omega)
    omega_hat_sq = torch.matmul(omega_hat, omega_hat)

    # Rotation: R = I + alpha * [ω]× + beta * [ω]×²
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(
        *omega.shape[:-1], 3, 3)
    R = (I + alpha[..., None] * omega_hat
         + beta[..., None] * omega_hat_sq)

    # Translation: t = V @ v, where V = I + beta * [ω]× + gamma * [ω]×²
    V = (I + beta[..., None] * omega_hat
         + gamma[..., None] * omega_hat_sq)
    t = torch.matmul(V, v.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # Assemble 4x4 homogeneous transform
    T = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_hat_torch(xi: torch.Tensor) -> torch.Tensor:
    """Batched hat map: se(3) vector → 4×4 matrix representation.

    Parameters
    ----------
    xi : (..., 6) = [omega(3), v(3)]

    Returns
    -------
    Xi : (..., 4, 4)
    """
    Xi = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    Xi[..., :3, :3] = skew_torch(xi[..., :3])
    Xi[..., :3, 3] = xi[..., 3:]
    return Xi


def se3_adjoint_torch(T: torch.Tensor) -> torch.Tensor:
    """Batched SE(3) adjoint Ad(T) of a pose, mapping body twist [ω, v]→[ω, v].

    Ad(R, t) = [[R,          0],
                [skew(t)·R,  R]]      (GTSAM [ω, v] convention).

    Parameters
    ----------
    T : (..., 4, 4) homogeneous transform.
    Returns
    -------
    Ad : (..., 6, 6)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    tR = torch.matmul(skew_torch(t), R)
    Z = torch.zeros_like(R)
    return torch.cat([torch.cat([R, Z], dim=-1),
                      torch.cat([tR, R], dim=-1)], dim=-2)


def so3_right_jacobian_torch(omega: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) right Jacobian J_r(ω) (3×3), robust at θ→0.

    J_r(ω) = I − (1−cosθ)/θ²·[ω]× + (θ−sinθ)/θ³·[ω]×²,   θ = |ω|.
    (The left Jacobian is J_r(−ω) = J_r(ω)ᵀ.)
    """
    theta = omega.norm(dim=-1, keepdim=True)
    theta_sq = theta ** 2
    safe = torch.where(theta < 1e-3, torch.ones_like(theta), theta)
    b = (1.0 - torch.cos(safe)) / safe ** 2
    c = (safe - torch.sin(safe)) / safe ** 3
    b = torch.where(theta < 1e-3, 0.5 - theta_sq / 24.0, b)
    c = torch.where(theta < 1e-3, 1.0 / 6.0 - theta_sq / 120.0, c)
    W = skew_torch(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(
        *omega.shape[:-1], 3, 3)
    return I - b[..., None] * W + c[..., None] * torch.matmul(W, W)


def se3_right_jacobian_torch(xi: torch.Tensor) -> torch.Tensor:
    """Batched SE(3) right Jacobian J_r(ξ) (6×6), ξ = [ω, v], robust at θ→0.

    J_r = [[J_r^SO3,  0      ],
           [Q,        J_r^SO3]]
    with the Barfoot coupling block

        Q(ω,v) = −½[v]× + c1·(P̂R̂ + R̂P̂ − P̂R̂P̂)
                       − c2·(P̂²R̂ + R̂P̂² − 3P̂R̂P̂)
                       + c3·(P̂R̂P̂² + P̂²R̂P̂),     P̂=[ω]×, R̂=[v]×

    c1=(θ−sinθ)/θ³, c2=(θ²+2cosθ−2)/2θ⁴, c3=(2θ−3sinθ+θcosθ)/2θ⁵.
    Verified against gtsam.Pose3.ExpmapDerivative and the ad-series to ~3e-16
    (and the small-θ Taylor branch to ≤2e-12).
    """
    omega = xi[..., :3]
    v = xi[..., 3:]
    Jso3 = so3_right_jacobian_torch(omega)

    theta = omega.norm(dim=-1, keepdim=True)
    theta_sq = theta ** 2
    safe = torch.where(theta < 1e-3, torch.ones_like(theta), theta)
    s, ct = torch.sin(safe), torch.cos(safe)
    c1 = (safe - s) / safe ** 3
    c2 = (safe ** 2 + 2.0 * ct - 2.0) / (2.0 * safe ** 4)
    c3 = (2.0 * safe - 3.0 * s + safe * ct) / (2.0 * safe ** 5)
    c1 = torch.where(theta < 1e-3, 1.0 / 6.0 - theta_sq / 120.0, c1)
    c2 = torch.where(theta < 1e-3, 1.0 / 24.0 - theta_sq / 720.0, c2)
    c3 = torch.where(theta < 1e-3, 1.0 / 120.0 - theta_sq / 2520.0, c3)

    P = skew_torch(omega)
    Rv = skew_torch(v)
    P2 = torch.matmul(P, P)
    PR, RP = torch.matmul(P, Rv), torch.matmul(Rv, P)
    PRP = torch.matmul(PR, P)
    Q = (-0.5 * Rv
         + c1[..., None] * (PR + RP - PRP)
         - c2[..., None] * (torch.matmul(P2, Rv) + torch.matmul(Rv, P2)
                            - 3.0 * PRP)
         + c3[..., None] * (torch.matmul(PR, P2) + torch.matmul(P2, RP)))
    Z = torch.zeros_like(Jso3)
    return torch.cat([torch.cat([Jso3, Z], dim=-1),
                      torch.cat([Q, Jso3], dim=-1)], dim=-2)
"""Differentiable kinematics shared by learned catheter models.

The functions in this module operate on physical piecewise-constant strain
directly.  They deliberately know nothing about PCA encoders, commanded base
joints, or a particular state-vector layout.
"""
from __future__ import annotations

from typing import Union

import torch

from .utils import se3_exp_torch


def pcs_forward_kinematics_torch(
    strain: torch.Tensor,
    interface_pose: torch.Tensor,
    section_lengths: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Propagate an unshearable/inextensible PCS rod from a full SE(3) pose.

    Parameters
    ----------
    strain:
        ``(..., n_sections, 3)`` angular strain ``[kappa_x,kappa_y,kappa_z]``.
    interface_pose:
        ``(..., 4, 4)`` pose of the proximal end of the modeled distal segment.
    section_lengths:
        Scalar or ``(n_sections,)`` material lengths.

    Returns
    -------
    ``(..., n_sections + 1, 4, 4)`` material-frame transforms.

    Notes
    -----
    The translational strain is fixed to local ``e3``.  This matches the native
    z-tangent convention used in ``cr_meta_lnn`` after SOFA convention conversion.
    """
    if strain.ndim < 2 or strain.shape[-1] != 3:
        raise ValueError("strain must have shape (..., n_sections, 3)")
    if interface_pose.shape[:-2] != strain.shape[:-2] or \
            interface_pose.shape[-2:] != (4, 4):
        raise ValueError(
            "interface_pose must have shape strain.shape[:-2] + (4, 4)")

    n_sections = strain.shape[-2]
    ds = torch.as_tensor(section_lengths, device=strain.device,
                         dtype=strain.dtype)
    if ds.ndim == 0:
        ds = ds.expand(n_sections)
    if ds.shape != (n_sections,):
        raise ValueError(
            f"section_lengths must be scalar or ({n_sections},), got {tuple(ds.shape)}")

    scaled_ds = ds.reshape(*([1] * (strain.ndim - 2)), n_sections, 1)
    omega = strain * scaled_ds
    translation = torch.zeros_like(omega)
    translation[..., 2] = ds
    increments = se3_exp_torch(torch.cat([omega, translation], dim=-1))

    frames = [interface_pose]
    current = interface_pose
    for section in range(n_sections):
        current = torch.matmul(current, increments[..., section, :, :])
        frames.append(current)
    return torch.stack(frames, dim=-3)


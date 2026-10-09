"""Stretch-reflex recruitment law mapping threshold lengths to excitations."""
import numpy as np


def lambda_to_excitation(lam, length_norm, velocity_norm, g_tonic=50.0, g_phasic=0.1):
    """Convert per-muscle threshold lengths into muscle excitations.

    ``e_i = clip( g_tonic * (length_norm_i - lam_i) + [g_phasic * velocity_norm_i]_+ , 0, 1 )``

    The tonic term is a signed length feedback about the threshold ``lam``; the
    phasic term is a rectified velocity feedback that adds drive only while the
    muscle is lengthening and never subtracts during shortening. Lengths and
    velocities are normalized by optimal fiber length (velocity positive when
    lengthening).

    Args:
        lam: per-muscle threshold length (the held command), shape ``(n_muscles,)``.
        length_norm: current normalized fiber length, same shape.
        velocity_norm: current normalized fiber velocity, same shape.
        g_tonic: tonic (length) reflex gain, shared by all muscles.
        g_phasic: phasic (velocity) reflex gain, shared by all muscles.

    Returns:
        Excitation in ``[0, 1]`` as ``float32``, same shape as the inputs.
    """
    lam = np.asarray(lam, dtype=np.float64)
    length_norm = np.asarray(length_norm, dtype=np.float64)
    velocity_norm = np.asarray(velocity_norm, dtype=np.float64)
    phasic = np.maximum(g_phasic * velocity_norm, 0.0)
    drive = g_tonic * (length_norm - lam) + phasic
    return np.clip(drive, 0.0, 1.0).astype(np.float32)

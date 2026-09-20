import numpy as np
import pytest

from skeleton_adapter.adapters import (
    HEAD,
    HIP,
    LELBOW,
    LFOOT,
    LHIP,
    LKNEE,
    LSHOULDER,
    LWRIST,
    NECK,
    RELBOW,
    RFOOT,
    RHIP,
    RKNEE,
    RSHOULDER,
    RWRIST,
    SPINE,
    THORAX,
    from_h36m,
    from_mediapipe,
    from_videopose3d,
    normalize_scale,
)
from skeleton_adapter.mediapipe_landmarks import MP
from skeleton_adapter.synthetic_poses import (
    t_pose_h36m_32,
    t_pose_mediapipe_33,
    t_pose_videopose3d_17,
)


def test_from_h36m_shape_and_selection():
    joints_32 = np.arange(32 * 3, dtype=float).reshape(32, 3)
    out = from_h36m(joints_32)
    assert out.shape == (17, 3)
    selected = joints_32[[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]]
    np.testing.assert_allclose(out, selected - selected[0])


def test_from_h36m_rejects_wrong_shape():
    with pytest.raises(ValueError):
        from_h36m(np.zeros((17, 3)))


def test_from_h36m_root_centered():
    out = from_h36m(t_pose_h36m_32())
    np.testing.assert_allclose(out[HIP], [0, 0, 0])


def test_from_videopose3d_centers_on_root():
    shifted = t_pose_videopose3d_17() + np.array([1000.0, 2000.0, 3000.0])
    out = from_videopose3d(shifted)
    np.testing.assert_allclose(out[HIP], [0, 0, 0])
    original = t_pose_videopose3d_17()
    np.testing.assert_allclose(out[HEAD] - out[HIP], original[HEAD] - original[HIP])


def test_from_videopose3d_rejects_wrong_shape():
    with pytest.raises(ValueError):
        from_videopose3d(np.zeros((32, 3)))


def test_h36m_and_videopose3d_agree_on_same_pose():
    np.testing.assert_allclose(
        from_h36m(t_pose_h36m_32()),
        from_videopose3d(t_pose_videopose3d_17()),
    )


def test_from_mediapipe_rejects_wrong_shape():
    with pytest.raises(ValueError):
        from_mediapipe(np.zeros((33, 2)))


def test_from_mediapipe_direct_mapped_joints():
    lm = t_pose_mediapipe_33()
    out = from_mediapipe(lm)
    pelvis_m = (lm[MP.LEFT_HIP] + lm[MP.RIGHT_HIP]) / 2
    np.testing.assert_allclose(out[RHIP], (lm[MP.RIGHT_HIP] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[LHIP], (lm[MP.LEFT_HIP] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[RKNEE], (lm[MP.RIGHT_KNEE] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[LKNEE], (lm[MP.LEFT_KNEE] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[RFOOT], (lm[MP.RIGHT_ANKLE] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[LFOOT], (lm[MP.LEFT_ANKLE] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[RSHOULDER], (lm[MP.RIGHT_SHOULDER] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[LSHOULDER], (lm[MP.LEFT_SHOULDER] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[RELBOW], (lm[MP.RIGHT_ELBOW] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[LELBOW], (lm[MP.LEFT_ELBOW] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[RWRIST], (lm[MP.RIGHT_WRIST] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[LWRIST], (lm[MP.LEFT_WRIST] - pelvis_m) * 1000)


def test_from_mediapipe_synthesized_joints():
    lm = t_pose_mediapipe_33()
    out = from_mediapipe(lm)
    pelvis_m = (lm[MP.LEFT_HIP] + lm[MP.RIGHT_HIP]) / 2
    thorax_m = (lm[MP.LEFT_SHOULDER] + lm[MP.RIGHT_SHOULDER]) / 2

    np.testing.assert_allclose(out[HIP], [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(out[THORAX], (thorax_m - pelvis_m) * 1000)
    np.testing.assert_allclose(out[SPINE], (out[HIP] + out[THORAX]) / 2)
    np.testing.assert_allclose(out[HEAD], (lm[MP.NOSE] - pelvis_m) * 1000)
    np.testing.assert_allclose(out[NECK], (out[THORAX] + out[HEAD]) / 2)


def test_from_mediapipe_matches_h36m_on_directly_mapped_joints():
    # Same T-pose in both raw formats can't agree everywhere (MediaPipe has
    # no real spine/neck markers, ours are synthesized), but every joint
    # MediaPipe actually measures directly should line up exactly.
    h36m_out = from_h36m(t_pose_h36m_32())
    mp_out = from_mediapipe(t_pose_mediapipe_33())
    directly_mapped = (
        RHIP, RKNEE, RFOOT, LHIP, LKNEE, LFOOT,
        LSHOULDER, LELBOW, LWRIST, RSHOULDER, RELBOW, RWRIST,
    )
    for idx in directly_mapped:
        np.testing.assert_allclose(h36m_out[idx], mp_out[idx], atol=1e-6)


def test_normalize_scale_sets_trunk_length_to_one():
    out = from_h36m(t_pose_h36m_32())
    scaled = normalize_scale(out)
    assert np.linalg.norm(scaled[THORAX] - scaled[HIP]) == pytest.approx(1.0)


def test_normalize_scale_rejects_zero_trunk_length():
    with pytest.raises(ValueError):
        normalize_scale(np.zeros((17, 3)))


def test_all_adapters_return_17x3():
    assert from_h36m(t_pose_h36m_32()).shape == (17, 3)
    assert from_videopose3d(t_pose_videopose3d_17()).shape == (17, 3)
    assert from_mediapipe(t_pose_mediapipe_33()).shape == (17, 3)

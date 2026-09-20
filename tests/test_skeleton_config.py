from skeleton_adapter.skeleton import SKELETON


def test_seventeen_joints():
    assert SKELETON.num_joints == 17
    assert len(SKELETON.joint_names) == 17


def test_joint_order_matches_videopose3d():
    assert SKELETON.joint_names == [
        "Hip", "RHip", "RKnee", "RFoot",
        "LHip", "LKnee", "LFoot",
        "Spine", "Thorax", "Neck/Nose", "Head",
        "LShoulder", "LElbow", "LWrist",
        "RShoulder", "RElbow", "RWrist",
    ]


def test_parent_hierarchy_matches_videopose3d():
    expected = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    assert SKELETON.parents.tolist() == expected


def test_root_has_no_parent():
    assert SKELETON.parents[SKELETON.root_joint] == -1


def test_left_right_joint_sets():
    assert SKELETON.joints_left == [4, 5, 6, 11, 12, 13]
    assert SKELETON.joints_right == [1, 2, 3, 14, 15, 16]
    assert set(SKELETON.joints_left).isdisjoint(SKELETON.joints_right)


def test_bones_exclude_root():
    bones = SKELETON.bones()
    assert len(bones) == 16
    assert all(child != SKELETON.root_joint for child, _ in bones)

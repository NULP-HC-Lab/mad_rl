OBSERVABLE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
CONTROLLABLE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

FOOT_BODY_NAMES = "(left|right)_ankle_roll"

LEFT_LEG_JOINT_INDICES = [0, 2, 4, 6, 8, 10]
RIGHT_LEG_JOINT_INDICES = [1, 3, 5, 7, 9, 11]
SYMMETRY_NEGATE_JOINT_INDICES = [2, 3, 4, 5, 10, 11]

HEIGHT_SCAN_SIZE = (0.4, 1.0)
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_NUM_ROWS = int(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION) + 1
HEIGHT_SCAN_NUM_COLS = int(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION) + 1

OBS_BASE_LIN_VEL_INDICES = slice(0, 3)
OBS_BASE_ANG_VEL_INDICES = slice(OBS_BASE_LIN_VEL_INDICES.stop, OBS_BASE_LIN_VEL_INDICES.stop + 3)
OBS_PROJECTED_GRAVITY_INDICES = slice(OBS_BASE_ANG_VEL_INDICES.stop, OBS_BASE_ANG_VEL_INDICES.stop + 3)
OBS_VELOCITY_COMMANDS_INDICES = slice(OBS_PROJECTED_GRAVITY_INDICES.stop, OBS_PROJECTED_GRAVITY_INDICES.stop + 3)
OBS_JOINT_POSITIONS_INDICES = slice(
    OBS_VELOCITY_COMMANDS_INDICES.stop,
    OBS_VELOCITY_COMMANDS_INDICES.stop + len(OBSERVABLE_JOINT_NAMES),
)
OBS_JOINT_VELOCITIES_INDICES = slice(
    OBS_JOINT_POSITIONS_INDICES.stop,
    OBS_JOINT_POSITIONS_INDICES.stop + len(OBSERVABLE_JOINT_NAMES),
)
OBS_PREVIOUS_ACTION_INDICES = slice(
    OBS_JOINT_VELOCITIES_INDICES.stop,
    OBS_JOINT_VELOCITIES_INDICES.stop + len(CONTROLLABLE_JOINT_NAMES),
)
OBS_HEIGHT_SCAN_INDICES = slice(
    OBS_PREVIOUS_ACTION_INDICES.stop,
    OBS_PREVIOUS_ACTION_INDICES.stop + HEIGHT_SCAN_NUM_ROWS * HEIGHT_SCAN_NUM_COLS,
)

# Full body joint ordering:
#   [ 0] left_hip_pitch_joint
#   [ 1] right_hip_pitch_joint
#   [ 2] waist_yaw_joint
#   [ 3] left_hip_roll_joint
#   [ 4] right_hip_roll_joint
#   [ 5] waist_roll_joint
#   [ 6] left_hip_yaw_joint
#   [ 7] right_hip_yaw_joint
#   [ 8] waist_pitch_joint
#   [ 9] left_knee_joint
#   [10] right_knee_joint
#   [11] left_shoulder_pitch_joint
#   [12] right_shoulder_pitch_joint
#   [13] left_ankle_pitch_joint
#   [14] right_ankle_pitch_joint
#   [15] left_shoulder_roll_joint
#   [16] right_shoulder_roll_joint
#   [17] left_ankle_roll_joint
#   [18] right_ankle_roll_joint
#   [19] left_shoulder_yaw_joint
#   [20] right_shoulder_yaw_joint
#   [21] left_elbow_joint
#   [22] right_elbow_joint
#   [23] left_wrist_roll_joint
#   [24] right_wrist_roll_joint
#   [25] left_wrist_pitch_joint
#   [26] right_wrist_pitch_joint
#   [27] left_wrist_yaw_joint
#   [28] right_wrist_yaw_joint
#   [29] index_0_joint
#   [30] middle_0_joint
#   [31] thumb_0_joint
#   [32] index_0_joint_0
#   [33] middle_0_joint_0
#   [34] thumb_0_joint_0
#   [35] index_1_joint
#   [36] middle_1_joint
#   [37] thumb_1_joint
#   [38] index_1_joint_0
#   [39] middle_1_joint_0
#   [40] thumb_1_joint_0
#   [41] thumb_2_joint
#   [42] thumb_2_joint_0

# Lower body joint ordering:
#   [ 0]  left_hip_pitch_joint      [ 1]  right_hip_pitch_joint
#   [ 2]  left_hip_roll_joint       [ 3]  right_hip_roll_joint
#   [ 4]  left_hip_yaw_joint        [ 5]  right_hip_yaw_joint
#   [ 6]  left_knee_joint           [ 7]  right_knee_joint
#   [ 8]  left_ankle_pitch_joint    [ 9]  right_ankle_pitch_joint
#   [10] left_ankle_roll_joint      [11] right_ankle_roll_joint

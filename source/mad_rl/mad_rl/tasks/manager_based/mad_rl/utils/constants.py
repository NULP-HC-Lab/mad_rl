# Regex patterns for controllable joints used in both observations and actions
CONTROLLABLE_JOINT_NAMES = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_.*",
]

FOOT_BODY_NAMES = "(left|right)_ankle_roll"

# Joint index mapping (12 joints total):
#   [0]  left_hip_pitch_joint       [1]  right_hip_pitch_joint
#   [2]  left_hip_roll_joint        [3]  right_hip_roll_joint
#   [4]  left_hip_yaw_joint         [5]  right_hip_yaw_joint
#   [6]  left_knee_joint            [7]  right_knee_joint
#   [8]  left_ankle_pitch_joint     [9]  right_ankle_pitch_joint
#   [10] left_ankle_roll_joint      [11] right_ankle_roll_joint
LEFT_LEG_JOINT_INDICES = [0, 2, 4, 6, 8, 10]
RIGHT_LEG_JOINT_INDICES = [1, 3, 5, 7, 9, 11]

# Joints that need sign negation for left-right symmetry (roll and yaw joints)
SYMMETRY_NEGATE_JOINT_INDICES = [2, 3, 4, 5, 10, 11]

# Observation tensor layout (48 dimensions total)
OBS_BASE_LIN_VEL_INDICES = slice(0, 3)  # Base linear velocity (x, y, z)
OBS_BASE_ANG_VEL_INDICES = slice(3, 6)  # Base angular velocity (roll, pitch, yaw)
OBS_PROJECTED_GRAVITY_INDICES = slice(6, 9)  # Projected gravity vector (x, y, z)
OBS_VELOCITY_COMMANDS_INDICES = slice(9, 12)  # Velocity commands (forward, lateral, yaw_rate)
OBS_JOINT_POSITIONS_INDICES = slice(12, 24)  # Joint positions (12 joints)
OBS_JOINT_VELOCITIES_INDICES = slice(24, 36)  # Joint velocities (12 joints)
OBS_PREVIOUS_ACTIONS_INDICES = slice(36, 48)  # Previous actions (12 joints)

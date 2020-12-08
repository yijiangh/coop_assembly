import datetime

# planning parameters
from .stiffness import TRANS_TOL, ROT_TOL
from .stream import ENABLE_SELF_COLLISIONS, IK_MAX_ATTEMPTS, PREGRASP_MAX_ATTEMPTS, GRASP_MAX_ATTEMPTS, \
    ALLOWABLE_BAR_COLLISION_DEPTH, EPSILON, ANGLE, POS_STEP_SIZE, ORI_STEP_SIZE, RETREAT_DISTANCE, MAX_DISTANCE, JOINT_JUMP_THRESHOLD
from .robot_setup import RESOLUTION

#################################################
# Logging

def config_specific_file_name(config, overwrite, tag=None, interfix='', suffix='.json'):
    date_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    file_name = '{}_{}_{}-{}{}{}{}'.format(config.problem, interfix,
        config.algorithm, config.bias,
        '_'+tag if tag is not None else '',
        '_'+date_time if not overwrite else '',
        suffix)
    return file_name

def get_global_parameters():
# from .stiffness import TRANS_TOL, ROT_TOL
# from .stream import ENABLE_SELF_COLLISIONS, IK_MAX_ATTEMPTS, PREGRASP_MAX_ATTEMPTS, GRASP_MAX_ATTEMPTS,
#     ALLOWABLE_BAR_COLLISION_DEPTH, EPSILON, ANGLE, POS_STEP_SIZE, ORI_STEP_SIZE, RETREAT_DISTANCE, MAX_DISTANCE, JOINT_JUMP_THRESHOLD
    return {
        'trans_tol': TRANS_TOL,
        'rot_tol': ROT_TOL,
        'joint_resolution': RESOLUTION,
        'joint_jump_threshold': JOINT_JUMP_THRESHOLD,
        'enable_self_collisions': ENABLE_SELF_COLLISIONS,
        'pos_step_size': POS_STEP_SIZE,
        'ori_step_size': ORI_STEP_SIZE,
        'retreat_distance': RETREAT_DISTANCE,
    }




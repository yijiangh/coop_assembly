import numpy as np
from termcolor import cprint

from pybullet_planning import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration, BLACK, RED, GREEN, remove_all_debug, apply_alpha, pairwise_collision, \
    set_color, refine_path, get_collision_fn, link_from_name, BASE_LINK
from coop_assembly.data_structure.utils import MotionTrajectory
from .stream import ENABLE_SELF_COLLISIONS
from .robot_setup import get_disabled_collisions, EE_LINK_NAME

##################################################

def validate_trajectories(element_from_index, fixed_obstacles, trajectories, allow_failure=False):
    if trajectories is None:
        return False
    # TODO: combine all validation procedures
    remove_all_debug()
    for element in element_from_index.values():
        set_color(element.body, np.zeros(4))

    print('Trajectories:', len(trajectories))
    obstacles = list(fixed_obstacles)
    for i, trajectory in enumerate(trajectories):
        robot = trajectory.robot
        joints = trajectory.joints
        attachments = trajectory.attachments
        extra_disabled_collisions = set()
        for attach in trajectory.attachments:
            extra_disabled_collisions.add(((robot, link_from_name(robot, EE_LINK_NAME)), (attach.child, BASE_LINK)))

        collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments,
                                        self_collisions=ENABLE_SELF_COLLISIONS,
                                        disabled_collisions=get_disabled_collisions(trajectory.robot),
                                        extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits={}, #get_custom_limits(robot),
                                        max_distance=0)

        if isinstance(trajectory, MotionTrajectory) and \
            (trajectory.tag == 'transit2place'):
            trajectory.refine(10)

        for conf in trajectory.iterate():
            # if any(pairwise_collision(trajectory.robot, body) for body in obstacles):
                # for attach in trajectory.attachments:
                #     set_color(attach.child, GREEN)
            if collision_fn(conf, diagnosis=has_gui()):
                print('Collision on trajectory {} | Element: {} | {}'.format(i, trajectory.element, trajectory.tag))
                if not allow_failure:
                    return False

        if isinstance(trajectory, MotionTrajectory)\
            and trajectory.tag == 'place_retreat':
            # set into goal pose
            body = element_from_index[trajectory.element].body
            set_color(body, apply_alpha(RED))
            obstacles.append(body)
    return True

import numpy as np
from termcolor import cprint

from pybullet_planning import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration, BLACK, RED, GREEN, BLUE, remove_all_debug, apply_alpha, pairwise_collision, \
    set_color, refine_path, get_collision_fn, link_from_name, BASE_LINK, get_links, wait_if_gui, set_pose
from coop_assembly.data_structure.utils import MotionTrajectory
from .stream import ENABLE_SELF_COLLISIONS
from .robot_setup import get_disabled_collisions, EE_LINK_NAME

##################################################

def validate_trajectories(element_from_index, fixed_obstacles, trajectories,
    grounded_elements={}, allow_failure=False, bar_only=False, refine_num=10):
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
            ee_link = link_from_name(robot, EE_LINK_NAME) if not bar_only else get_links(robot)[-1]
            extra_disabled_collisions.add(((robot, ee_link), (attach.child, BASE_LINK)))
        disabled_collisions = {} if bar_only else get_disabled_collisions(trajectory.robot)
        collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments,
                                        self_collisions=ENABLE_SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits={}, #get_custom_limits(robot),
                                        max_distance=0)

        # if isinstance(trajectory, MotionTrajectory) and \
        #     (trajectory.tag == 'transit2place'):
        #     trajectory.refine(refine_num)

        path = list(trajectory.iterate())
        for t, conf in enumerate(path):
            # if any(pairwise_collision(trajectory.robot, body) for body in obstacles):
                # for attach in trajectory.attachments:
                #     set_color(attach.child, GREEN)
            if trajectory.tag == 'place_approach' and t > len(path)-5:
                continue
            if collision_fn(conf, diagnosis=has_gui()):
                print('Collision on trajectory {}-#{}/{} | Element: {} | {}'.format(i, t, len(path), trajectory.element, trajectory.tag))
                if not allow_failure:
                    return False

        if isinstance(trajectory, MotionTrajectory) \
            and ((not bar_only and trajectory.tag == 'place_retreat') or (bar_only and trajectory.tag == 'place_approach')):
            # set into goal pose
            body = element_from_index[trajectory.element].body
            set_pose(body, element_from_index[trajectory.element].goal_pose.value)
            if trajectory.element in grounded_elements:
                set_color(body, apply_alpha(BLACK))
            else:
                set_color(body, apply_alpha(BLUE))
            obstacles.append(body)
        # wait_if_gui()
    return True

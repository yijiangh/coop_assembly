import numpy as np
from termcolor import cprint

from pybullet_planning import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration, BLACK, RED, GREEN, BLUE, remove_all_debug, apply_alpha, pairwise_collision, \
    set_color, refine_path, get_collision_fn, link_from_name, BASE_LINK, get_links, wait_if_gui, set_pose
from coop_assembly.data_structure.utils import MotionTrajectory
from .stream import ENABLE_SELF_COLLISIONS, get_element_body_in_goal_pose, command_collision
from .robot_setup import get_disabled_collisions, EE_LINK_NAME
from .utils import recover_sequence, Command
from .visualization import label_elements

##################################################

def validate_trajectories(element_from_index, fixed_obstacles, trajectories,
    grounded_elements={}, allow_failure=False, bar_only=False, refine_num=10, watch=False):
    if trajectories is None:
        return False
    # TODO: combine all validation procedures
    remove_all_debug()
    label_elements({e:element_from_index[e].body for e in element_from_index}, body_index=False)
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
            set_color(attach.child, GREEN)
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
        # TODO
        # trajectory.refine(refine_num)

        valid = True
        path = list(trajectory.iterate())
        for t, conf in enumerate(path):
            # if any(pairwise_collision(trajectory.robot, body) for body in obstacles):
            # for attach in trajectory.attachments:
            #     set_color(attach.child, GREEN)
            if collision_fn(conf, diagnosis=has_gui()):
                # if trajectory.tag == 'place_approach' and \
                #     not bar_only and len(path) >= 2 and t > len(path)-2:
                #         pass
                # else:
                cprint('Collision on trajectory {}-#{}/{} | Element: {} | {}'.format(i, t, len(path), trajectory.element, trajectory.tag), 'red')
                valid = False
                if not allow_failure:
                    return False
            if watch:
                print('Traj {}-#{}/{} | Element: {} | {}'.format(i, t, len(path), trajectory.element, trajectory.tag))
                wait_if_gui()

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
    return valid

##############################################

def validate_pddl_plan(trajectories, bar_struct, fixed_obstacles, allow_failure=False, watch=False, **kwargs):
    element_from_index = bar_struct.get_element_from_index()
    element_seq = recover_sequence(trajectories, element_from_index)
    for traj in trajectories:
        if traj.element is not None and traj.tag=='place_approach':
            print('E{} : future E{}'.format(traj.element, element_seq[element_seq.index(traj.element):]))
            command = Command([traj])
            elements_order = [e for e in element_from_index if (e != traj.element)]
            bodies_order = get_element_body_in_goal_pose(element_from_index, elements_order)
            colliding = command_collision(command, bodies_order)
            for element2, unsafe in zip(elements_order, colliding):
                if unsafe:
                    command.set_unsafe(element2)
                else:
                    command.set_safe(element2)
            facts = [('Collision', command, e2) for e2 in command.colliding]
            print('Collision facts: ', command.colliding)
            # * checking (forall (?e2) (imply (Collision ?t ?e2) (Removed ?e2)))
            valid = set(command.colliding) <= set(element_seq[element_seq.index(traj.element):])
            if not valid:
                if not allow_failure:
                    return False
                cprint('Collision facts violated!', 'red')
                wait_if_gui()
        print('------------')
    valid = validate_trajectories(element_from_index, fixed_obstacles, trajectories, \
        grounded_elements=bar_struct.get_grounded_bar_keys(), allow_failure=allow_failure, watch=watch, **kwargs)
    cprint('Valid: {}'.format(valid), 'green' if valid else 'red')
    return valid

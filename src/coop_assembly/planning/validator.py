import numpy as np
from termcolor import cprint

from pybullet_planning import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration, BLACK, RED, GREEN, BLUE, remove_all_debug, apply_alpha, pairwise_collision, \
    set_color, refine_path, get_collision_fn, link_from_name, BASE_LINK, get_links, wait_if_gui, set_pose, pairwise_link_collision_info, \
    draw_collision_diagnosis
from coop_assembly.data_structure.utils import MotionTrajectory
from .stream import ENABLE_SELF_COLLISIONS, get_element_body_in_goal_pose, command_collision, MAX_DISTANCE, ALLOWABLE_BAR_COLLISION_DEPTH
from .robot_setup import get_disabled_collisions, EE_LINK_NAME
from .utils import recover_sequence, Command, get_index_from_bodies
from .visualization import label_elements, visualize_collision_digraph
from .stiffness import evaluate_stiffness, create_stiffness_checker

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
    valid = True
    obstacles = list(fixed_obstacles)
    for i, trajectory in enumerate(trajectories):
        cprint(trajectory, 'cyan')

        robot = trajectory.robot
        joints = trajectory.joints
        attachments = trajectory.attachments
        extra_disabled_collisions = set()
        for attach in trajectory.attachments:
            set_color(attach.child, GREEN)
            ee_link = link_from_name(robot, EE_LINK_NAME) if not bar_only else get_links(robot)[-1]
            extra_disabled_collisions.add(((robot, ee_link), (attach.child, BASE_LINK)))
        # if detach, ignore end effector's collision with the element
        if 'retreat' in trajectory.tag:
            extra_disabled_collisions.add(((robot, ee_link), (element_from_index[trajectory.element].body, BASE_LINK)))

        disabled_collisions = {} if bar_only else get_disabled_collisions(trajectory.robot)
        collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments,
                                        self_collisions=True, # ENABLE_SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits={}, #get_custom_limits(robot),
                                        max_distance=MAX_DISTANCE)

        # if isinstance(trajectory, MotionTrajectory) and \
        #     (trajectory.tag == 'transit2place'):
        #     trajectory.refine(refine_num)
        # TODO
        # trajectory.refine(refine_num)

        path = list(trajectory.iterate())
        for t, conf in enumerate(path):
            # if any(pairwise_collision(trajectory.robot, body) for body in obstacles):
            # for attach in trajectory.attachments:
            #     set_color(attach.child, GREEN)
            if collision_fn(conf, diagnosis=has_gui()):
                bar_in_collision = False
                for at in attachments:
                    for obstacle in obstacles:
                        cr = pairwise_link_collision_info(at.child, BASE_LINK, obstacle, BASE_LINK)
                        penetration_depth = draw_collision_diagnosis(cr)
                        # print('depth #{}-#{}: {}'.format(at.child, obstacle, penetration_depth))
                        if penetration_depth is not None and penetration_depth > ALLOWABLE_BAR_COLLISION_DEPTH:
                            bar_in_collision = True
                if bar_in_collision:
                    cprint('Collision on trajectory {}-#{}/{} | Element: {} | {}'.format(i, t, len(path), trajectory.element, trajectory.tag), 'red')

                    valid = False
                    if not allow_failure:
                        return False
                else:
                    cprint('Bar collision under tolerance {}, viewed as a valid solution.'.format(ALLOWABLE_BAR_COLLISION_DEPTH), 'yellow')

        if isinstance(trajectory, MotionTrajectory) \
            and 'retreat' in trajectory.tag:
            # set into goal pose
            body = element_from_index[trajectory.element].body
            set_pose(body, element_from_index[trajectory.element].goal_pose.value)
            if trajectory.element in grounded_elements:
                set_color(body, apply_alpha(BLACK))
            else:
                set_color(body, apply_alpha(BLUE))
            obstacles.append(body)
        # wait_if_gui()
        cprint('>'*10, 'cyan')
    return valid

##############################################

def validate_pddl_plan(trajectories, fixed_obstacles, element_from_index, grounded_elements, allow_failure=False, watch=False, debug=False, **kwargs):
    print('Collided element should be included in the future (unprinted) set.')
    label_elements({e:element_from_index[e].body for e in element_from_index}, body_index=True)
    element_seq = recover_sequence(trajectories, element_from_index)
    index_from_bodies = get_index_from_bodies(element_from_index)
    collision_facts = []
    for traj in trajectories:
        if traj.element is not None and traj.tag=='place_approach':
            print('E{} : future E{}'.format(traj.element, element_seq[element_seq.index(traj.element):]))
            command = Command([traj])
            elements_order = [e for e in element_from_index if (e != traj.element)]
            bodies_order = get_element_body_in_goal_pose(element_from_index, elements_order)
            colliding = command_collision(command, bodies_order, index_from_bodies=index_from_bodies, debug=False)
            for element2, unsafe in zip(elements_order, colliding):
                if unsafe:
                    command.set_unsafe(element2)
                else:
                    command.set_safe(element2)
            facts = [('Collision', command, e2) for e2 in command.colliding]
            print('Collision facts: ', command.colliding)
            collision_facts.extend(facts)
            # * checking (forall (?e2) (imply (Collision ?t ?e2) (Removed ?e2)))
            valid = set(command.colliding) <= set(element_seq[element_seq.index(traj.element):])
            if not valid:
                if not allow_failure:
                    return False
                cprint('Collision facts violated!', 'red')
                wait_if_gui()
        print('------------')

    # * visualize the collision constraint directional graph
    # https://networkx.github.io/documentation/stable/tutorial.html#directed-graphs
    if debug:
        visualize_collision_digraph(collision_facts)

    valid = validate_trajectories(element_from_index, fixed_obstacles, trajectories, \
        grounded_elements=grounded_elements, allow_failure=allow_failure, watch=watch, **kwargs)
    return valid

def compute_plan_deformation(bar_struct, plan):
    checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct, verbose=False)
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    if plan is None:
        return trans_tol, rot_tol

    printed = []
    translations = []
    rotations = []
    for element in plan:
        printed.append(element)
        deformation = evaluate_stiffness(bar_struct, printed,
                                         checker=checker, fem_element_from_bar_id=fem_element_from_bar_id, verbose=True)
        trans, rot, _, _ = checker.get_max_nodal_deformation()
        translations.append([bool(deformation.success), trans])
        rotations.append(rot)
    # TODO: could return full history
    return translations

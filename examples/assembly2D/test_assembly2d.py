import pytest

from pybullet_planning import connect, has_gui, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose, get_date, RED, GREEN, refine_path, joints_from_names, \
    set_joint_positions, create_attachment, wait_if_gui, apply_alpha, get_movable_joints, get_links, invert

from coop_assembly.planning.utils import flatten_commands
from coop_assembly.planning.visualization import display_trajectories

from .stream import get_delta_pose_generator, get_2d_element_grasp_gen_fn, get_2d_pregrasp_gen_fn, compute_2d_place_path, get_2d_place_gen_fn, \
    xz_values_from_pose
from .parsing import parse_2D_truss
from .robot_setup import load_2d_world
# from .robot_setup import EE_FROM_TOOL

@pytest.mark.model
def test_import_model(viewer, problem, debug_mode):
    end_effector, floor, tool_from_ee = load_2d_world(viewer=viewer)
    element_from_index, connectors, grounded_elements = parse_2D_truss(problem, debug=debug_mode)
    wait_if_gui()

@pytest.mark.stream
def test_stream(viewer, problem, collision):
    end_effector, floor, tool_from_ee = load_2d_world(viewer=viewer)
    element_from_index, connectors, grounded_elements = parse_2D_truss(problem)
    wait_if_gui("Model loaded.")

    printed = set([(13,8), (8,4)])
    chosen = (6,4)

    # color_structure(element_bodies, printed, next_element=chosen, built_alpha=0.6)
    # wait_if_gui()

    n_attempts = 5
    # tool_pose = Pose(euler=Euler(yaw=np.pi/2))
    obstacles = [floor]

    # goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen_fn = get_2d_element_grasp_gen_fn(element_from_index, tool_from_ee, reverse_grasp=False,
        safety_margin_length=0.005)
    pregrasp_gen_fn = get_2d_pregrasp_gen_fn(element_from_index, obstacles, collision=collision, teleops=False)
    place_gen = get_2d_place_gen_fn(end_effector, tool_from_ee, element_from_index, obstacles,
        collisions=collision, verbose=True) #max_attempts=n_attempts,

    ee_joints = get_movable_joints(end_effector)
    ee_body_link = get_links(end_effector)[-1]

    # body_pose = element_from_index[chosen].goal_pose.value
    for _ in range(n_attempts):
        handles = []
        # * sample goal pose and grasp
        # couple rotations in goal pose' symmetry and translational grasp
        grasp, = next(grasp_gen_fn(chosen))
        # world_pose, = next(goal_pose_gen_fn(chosen))
        world_pose = element_from_index[chosen].goal_pose
        pregrasp_poses, = next(pregrasp_gen_fn(chosen, world_pose, printed=printed))

        # visualize grasp
        p = world_pose.value
        gripper_from_bar = grasp.attach
        set_pose(element_from_index[chosen].body, p)
        world_from_ee = end_effector_from_body(p, gripper_from_bar)

        ee_pose = xz_values_from_pose(world_from_ee)
        set_joint_positions(end_effector, ee_joints, ee_pose)

        attachment = create_attachment(end_effector, ee_body_link, element_from_index[chosen].body)
        # set_pose(end_effector, world_from_ee)

        handles.extend(draw_pose(world_from_ee, length=0.05))
        handles.extend(draw_pose(p, length=0.05))

        # * sample pick trajectory
        command, = next(place_gen(chosen, printed=printed, diagnosis=False))
        # command = compute_2d_place_path(end_effector, pregrasp_poses, grasp, chosen, element_from_index)
        # command = None

        print('Colliding: {}'.format(command.colliding))
        if not command:
            print('no command found')
            gripper_from_bar = grasp.attach
            for p in pregrasp_poses:
                # set_pose(element_from_index[chosen].body, p)
                world_from_ee = end_effector_from_body(p, gripper_from_bar)

                # set_pose(end_effector, world_from_ee)
                ee_pose = xz_values_from_pose(world_from_ee)
                set_joint_positions(end_effector, ee_joints, ee_pose)
                attachment.assign()

                handles.extend(draw_pose(world_from_ee, length=0.01))
                wait_if_gui()
            print('-'*10)
        else:
            print('command found!')
            trajs = flatten_commands([command])
            # time_step = None if has_gui() else 0.1
            time_step = 0.05
            display_trajectories(trajs, time_step=time_step, #video=True,
                                 animate=True, element_from_index=element_from_index)
            print('*'*10)

        wait_if_gui()
        remove_handles(handles)

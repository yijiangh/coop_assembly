import time
import numpy as np
from pybullet_planning import get_movable_joints, link_from_name, set_pose, \
    multiply, invert, inverse_kinematics, plan_direct_joint_motion, Attachment, set_joint_positions, plan_joint_motion, \
    get_configuration, wait_for_user, point_from_pose, HideOutput, load_pybullet, draw_pose, unit_quat, create_obj, \
    add_body_name, get_pose, pose_from_tform, connect, WorldSaver, get_sample_fn, \
    wait_for_duration, enable_gravity, enable_real_time, trajectory_controller, simulate_controller, \
    add_fixed_constraint, remove_fixed_constraint, Pose, Euler, get_collision_fn, LockRenderer, user_input, GREEN, BLUE, set_color
from coop_assembly.data_structure.utils import MotionTrajectory

##################################################

def step_trajectory(trajectory, attachments={}, time_step=np.inf):
    for _ in trajectory.iterate():
        for attachment in attachments:
            attachment.assign()
        if time_step == np.inf:
            wait_for_user()
        else:
            wait_for_duration(time_step)

def step_plan(plan, **kwargs):
    wait_for_user()
    attachments = {}
    for action, args in plan:
        trajectory = args[-1]
        if action == 'move':
            step_trajectory(trajectory, attachments, **kwargs)
        elif action == 'pick':
            attachment = trajectory.attachments.pop()
            step_trajectory(trajectory, attachments, **kwargs)
            attachments[attachment.child] = attachment
            step_trajectory(trajectory.reverse(), attachments, **kwargs)
        elif action == 'place':
            attachment = trajectory.attachments.pop()
            step_trajectory(trajectory, attachments, **kwargs)
            del attachments[attachment.child]
            step_trajectory(trajectory.reverse(), attachments, **kwargs)
        else:
            raise NotImplementedError(action)
    wait_for_user()

###################################

def display_trajectories(trajectories, animate=True, time_step=0.02, video=False):
    # node_points, ground_nodes,
    if trajectories is None:
        return
    # set_extrusion_camera(node_points)
    # planned_elements = recover_sequence(trajectories)
    # colors = sample_colors(len(planned_elements))
    # if not animate:
    #     draw_ordered(planned_elements, node_points)
    #     wait_for_user()
    #     disconnect()
    #     return

    # video_saver = None
    # if video:
    #     handles = draw_model(planned_elements, node_points, ground_nodes) # Allows user to adjust the camera
    #     wait_for_user()
    #     remove_all_debug()
    #     wait_for_duration(0.1)
    #     video_saver = VideoSaver('video.mp4') # has_gui()
    #     time_step = 0.001
    # else:
    #     wait_for_user()

    #element_bodies = dict(zip(planned_elements, create_elements(node_points, planned_elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected_nodes = set(ground_nodes)
    printed_elements = []
    print('Trajectories:', len(trajectories))
    for i, trajectory in enumerate(trajectories):
        #wait_for_user()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []

        if isinstance(trajectory, MotionTrajectory):
            for attach in trajectory.attachments:
                set_color(attach.child, GREEN)

        for _ in trajectory.iterate():
            # TODO: the robot body could be different
            # if isinstance(trajectory, PrintTrajectory):
            #     current_point = point_from_pose(trajectory.end_effector.get_tool_pose())
            #     if last_point is not None:
            #         # color = BLUE if is_ground(trajectory.element, ground_nodes) else RED
            #         color = colors[len(printed_elements)]
            #         handles.append(add_line(last_point, current_point, color=color, width=LINE_WIDTH))
            #     last_point = current_point

            if time_step is None:
                wait_for_user()
            else:
                wait_for_duration(time_step)

        if isinstance(trajectory, MotionTrajectory):
            for attach in trajectory.attachments:
                set_color(attach.child, BLUE)

        # if isinstance(trajectory, PrintTrajectory):
        #     if not trajectory.path:
        #         color = colors[len(printed_elements)]
        #         handles.append(draw_element(node_points, trajectory.element, color=color))
        #         #wait_for_user()
        #     is_connected = (trajectory.n1 in connected_nodes) # and (trajectory.n2 in connected_nodes)
        #     print('{}) {:9} | Connected: {} | Ground: {} | Length: {}'.format(
        #         i, str(trajectory), is_connected, is_ground(trajectory.element, ground_nodes), len(trajectory.path)))
        #     if not is_connected:
        #         wait_for_user()
        #     connected_nodes.add(trajectory.n2)
        #     printed_elements.append(trajectory.element)

    # if video_saver is not None:
    #     video_saver.restore()
    wait_for_user()

##################################################

def simulate_trajectory(trajectory, time_step=0.0):
    start_time = time.time()
    for sim_time in simulate_controller(trajectory_controller(trajectory.robot, trajectory.joints, trajectory.path)):
        if time_step:
            time.sleep(time_step)
        #print(sim_time, elapsed_time(start_time))

def simulate_plan(plan, time_step=0.0, real_time=False): #time_step=np.inf
    wait_for_user()
    enable_gravity()
    if real_time:
        enable_real_time()
    for action, args in plan:
        trajectory = args[-1]
        if action == 'move':
            simulate_trajectory(trajectory, time_step)
        elif action == 'pick':
            attachment = trajectory.attachments.pop()
            simulate_trajectory(trajectory, time_step)
            add_fixed_constraint(attachment.child, attachment.parent, attachment.parent_link)
            simulate_trajectory(trajectory.reverse(), time_step)
        elif action == 'place':
            ttachment = trajectory.attachments.pop()
            simulate_trajectory(trajectory, time_step)
            remove_fixed_constraint(attachment.child, attachment.parent, attachment.parent_link)
            simulate_trajectory(trajectory.reverse(), time_step)
        else:
            raise NotImplementedError(action)
    wait_for_user()

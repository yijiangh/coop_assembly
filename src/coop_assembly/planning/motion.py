import time
import numpy as np
from pybullet_planning import get_movable_joints, link_from_name, set_pose, \
    multiply, invert, inverse_kinematics, plan_direct_joint_motion, Attachment, set_joint_positions, plan_joint_motion, \
    get_configuration, wait_for_user, point_from_pose, HideOutput, load_pybullet, draw_pose, unit_quat, create_obj, \
    add_body_name, get_pose, pose_from_tform, connect, WorldSaver, get_sample_fn, \
    wait_for_duration, enable_gravity, enable_real_time, trajectory_controller, simulate_controller, \
    add_fixed_constraint, remove_fixed_constraint, Pose, Euler, get_collision_fn, LockRenderer, user_input

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

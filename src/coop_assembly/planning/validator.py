import numpy as np

from pybullet_planning import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration, BLACK, RED, remove_all_debug, apply_alpha, pairwise_collision, \
    set_color
from coop_assembly.data_structure.utils import MotionTrajectory

##################################################

def validate_trajectories(element_bodies, fixed_obstacles, trajectories):
    if trajectories is None:
        return False
    # TODO: combine all validation procedures
    remove_all_debug()
    for body in element_bodies.values():
        set_color(body, np.zeros(4))

    print('Trajectories:', len(trajectories))
    obstacles = list(fixed_obstacles)
    for i, trajectory in enumerate(trajectories):
        for _ in trajectory.iterate():
            #wait_for_user()
            if any(pairwise_collision(trajectory.robot, body) for body in obstacles):
                if has_gui():
                    print('Collision on trajectory {}'.format(i))
                    wait_for_user()
                return False
            # TODO Sweep checking near pregrasp pose

        if isinstance(trajectory, MotionTrajectory):
            body = element_bodies[trajectory.element]
            set_color(body, apply_alpha(RED))
            obstacles.append(body)
    return True

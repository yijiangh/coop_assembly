import numpy as np
from collections import defaultdict, deque

# from pddlstream.utils import get_connected_components
from pybullet_planning import HideOutput, load_pybullet, set_static, set_joint_positions, joints_from_names, \
    create_plane, set_point, Point, link_from_name, get_link_pose, BodySaver, has_gui, wait_for_user, randomize, pairwise_link_collision, \
    BASE_LINK, is_connected, connect

from coop_assembly.data_structure.utils import MotionTrajectory
from coop_assembly.help_functions.shared_const import METER_SCALE
from .robot_setup import get_picknplace_robot_data, BUILT_PLATE_Z, EE_LINK_NAME, INITIAL_CONF
from .visualization import GROUND_COLOR, BACKGROUND_COLOR, SHADOWS

def load_world(use_floor=True, built_plate_z=BUILT_PLATE_Z, viewer=False):
    if not is_connected():
        connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)

    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, _, _, joint_names, _ = robot_data

    print('URDF: ', robot_urdf)
    obstacles = []
    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
        # 1/velocity = weight
        # print([get_max_velocity(robot, joint) for joint in get_movable_joints(robot)])
        set_static(robot)
        set_joint_positions(robot, joints_from_names(robot, joint_names), INITIAL_CONF)
        if use_floor:
            floor = create_plane(color=GROUND_COLOR)
            obstacles.append(floor)
            set_point(floor, Point(x=1.2, z=built_plate_z))
        else:
            floor = None
    return obstacles, robot

##################################################

def get_index_from_bodies(element_from_index):
    return {element_from_index[e].body : element_from_index[e].index for e in element_from_index}

##################################################

def recover_sequence(plan, element_from_index):
    if plan is None:
        return plan
    return [traj.element for traj in plan if isinstance(traj, MotionTrajectory) and traj.tag == 'place_approach']

def flatten_commands(commands):
    if commands is None:
        return None
    return [traj for command in commands for traj in command.trajectories]

##################################################

def prune_dominated(trajectories):
    # prune trajectory with more collision elements
    start_len = len(trajectories)
    for traj1 in list(trajectories):
        if any((traj1 != traj2) and (traj2.colliding <= traj1.colliding)
               for traj2 in trajectories):
            trajectories.remove(traj1)
    return len(trajectories) == start_len

##################################################

class Command(object):
    def __init__(self, trajectories=[], safe_per_element={}):
        self.trajectories = list(trajectories)
        self.safe_per_element = dict(safe_per_element)
        self.colliding = set()
    # @property
    # def print_trajectory(self):
    #     for traj in self.trajectories:
    #         if isinstance(traj, PrintTrajectory):
    #             return traj
    #     return None
    @property
    def start_conf(self):
        return self.trajectories[0].start_conf
    @property
    def end_conf(self):
        return self.trajectories[-1].end_conf
    @property
    def start_robot(self):
        return self.trajectories[0].robot
    @property
    def end_robot(self):
        return self.trajectories[-1].robot
    # @property
    # def elements(self):
    #     return recover_sequence(self.trajectories)
    # @property
    # def directed_elements(self):
    #     return recover_directed_sequence(self.trajectories)
    def get_distance(self):
        return sum(traj.get_distance() for traj in self.trajectories)
    def get_link_distance(self, **kwargs):
        return sum(traj.get_link_distance(**kwargs) for traj in self.trajectories)
    def set_safe(self, element):
        assert self.safe_per_element.get(element, True) is True
        self.safe_per_element[element] = True
    def set_unsafe(self, element):
        assert self.safe_per_element.get(element, False) is False
        self.safe_per_element[element] = False
        self.colliding.add(element)
    def update_safe(self, elements):
        for element in elements:
            self.set_safe(element)
    def is_safe(self, elements, element_bodies):
        # TODO: check the end-effector first
        known_elements = set(self.safe_per_element) & set(elements)
        if not all(self.safe_per_element[e] for e in known_elements):
            return False
        unknown_elements = randomize(set(elements) - known_elements)
        if not unknown_elements:
            return True
        for trajectory in randomize(self.trajectories): # TODO: could cache each individual collision
            intersecting = trajectory.get_intersecting()
            for i in randomize(range(len(trajectory))):
                set_joint_positions(trajectory.robot, trajectory.joints, trajectory.path[i])
                for element in unknown_elements:
                    body = element_bodies[element]
                    #if not pairwise_collision(trajectory.robot, body):
                    #    self.set_unsafe(element)
                    #    return False
                    for robot_link, bodies in intersecting[i].items():
                        #print(robot_link, bodies, len(bodies))
                        if (element_bodies[element] in bodies) and pairwise_link_collision(
                                trajectory.robot, robot_link, body, link2=BASE_LINK):
                            self.set_unsafe(element)
                            return False
        self.update_safe(elements)
        return True
    def reverse(self):
        return self.__class__([traj.reverse() for traj in reversed(self.trajectories)],
                              safe_per_element=self.safe_per_element)
    def iterate(self):
        for trajectory in self.trajectories:
            for output in trajectory.iterate():
                yield output
    @property
    def start_time(self):
        return self.trajectories[0].start_time
    @property
    def end_time(self):
        return self.trajectories[-1].end_time
    @property
    def duration(self):
        return self.end_time - self.start_time
    def retime(self, start_time=0, **kwargs):
        for traj in self.trajectories:
            traj.retime(start_time=start_time)
            start_time += traj.duration
    def __repr__(self):
        return 'c[{}]'.format(','.join(map(repr, self.trajectories)))

##################################################

def get_connector_from_elements(connectors, elements):
    connector_from_elements = defaultdict(set)
    for e in elements:
        for c in connectors:
            if e in c:
                connector_from_elements[e].add(c)
    return connector_from_elements

def get_element_neighbors(connectors, elements):
    connector_from_elements = get_connector_from_elements(connectors, elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        for c in connector_from_elements[e]:
            element_neighbors[e].update(c)
        element_neighbors[e].remove(e)
        #     print('e: ', e)
        #     print('neighbors: ', element_neighbors[e])
        #     print('connector: ', connector_from_elements[e])
        #     input('floating element!!!')
    return element_neighbors

##################################################

def check_connected(connectors, grounded_elements, printed_elements):
    """check if a given partial structure is connected to the ground

    Parameters
    ----------
    connectors : list of 2-int tuples
        each entry are the indices into the element set,
    grounded_elements : set
        grounded element ids
    printed_elements : set
        printed element ids

    Returns
    -------
    bool
        True if connected to ground
    """
    # TODO: for stability might need to check 2-connected
    if not printed_elements:
        return True
    printed_grounded_elements = set(grounded_elements) & printed_elements
    if not printed_grounded_elements:
        return False
    element_neighbors = get_element_neighbors(connectors, printed_elements)
    queue = deque(printed_grounded_elements)
    visited_elements = set()
    while queue:
        n_element = queue.popleft()
        visited_elements.add(n_element)
        for element in element_neighbors[n_element]:
            if element in printed_elements and element not in visited_elements:
                visited_elements.add(element)
                queue.append(element)
    return printed_elements <= visited_elements

def get_connected_structures(connectors, elements):
    edges = {(e1, e2) for e1, neighbors in get_element_neighbors(connectors, elements).items()
             for e2 in neighbors}
    return get_connected_components(elements, edges)

######################################################

def get_midpoint(element_from_index, element):
    return np.average([element_from_index[element].axis_endpoints[i] for i in range(2)], axis=0)

def compute_z_distance(element_from_index, element):
    # Distance to a ground plane
    # Opposing gravitational force
    return get_midpoint(element_from_index, element)[2]

from collections import namedtuple
import numpy as np

from pybullet_planning import point_from_pose
from pybullet_planning import HideOutput, load_pybullet, set_static, set_joint_positions, joints_from_names, \
    create_plane, set_point, Point, link_from_name, get_link_pose, BodySaver, refine_path, set_pose, create_attachment
from coop_assembly.planning.robot_setup import TOOL_LINK_NAME

##################################################

""" simple container for an element
"""
Element = namedtuple('Element', ['index', 'axis_endpoints', 'radius', 'body', 'initial_pose', 'goal_pose',
                                 'grasps', 'goal_supports', 'layer'])

class WorldPose(object):
    """indexed pose in the world coordinate system
    """
    def __init__(self, index, value):
        self.index = index
        self.value = value
    def to_data(self):
        data = {'index' : self.index,
                'value' : [list(self.value[i]) for i in range(2)]}
        return data
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.index,
                                  str(np.array(point_from_pose(self.value))))

class Grasp(object):
    def __init__(self, index, num, approach, attach, retreat):
        self.index = index # bar vertex key
        self.num = num # grasp id associated for a bar
        self.approach = approach
        self.attach = attach
        self.retreat = retreat
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.index, self.num)

##################################################

class Trajectory(object):
    def __init__(self, robot, joints, path):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.path_from_link = {}
    @property
    def start_conf(self):
        if not self.path:
            return None
        return self.path[0]
    @property
    def end_conf(self):
        if not self.path:
            return None
        return self.path[-1]
    def get_link_path(self, link_name):
        link = link_from_name(self.robot, link_name)
        if link not in self.path_from_link:
            with BodySaver(self.robot):
                self.path_from_link[link] = []
                for conf in self.path:
                    set_joint_positions(self.robot, self.joints, conf)
                    self.path_from_link[link].append(get_link_pose(self.robot, link))
        return self.path_from_link[link]
    def reverse(self):
        raise NotImplementedError()
    def iterate(self):
        for conf in self.path:
            set_joint_positions(self.robot, self.joints, conf)
            yield conf
    def refine(self, num_steps):
       self.path = refine_path(self.robot, self.joints, self.path, num_steps)

class MotionTrajectory(Trajectory):
    def __init__(self, robot, joints, path, attachments=[], element=None, tag=None):
        super(MotionTrajectory, self).__init__(robot, joints, path)
        self.attachments = attachments
        self.element = element
        self.tag = tag
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1], self.attachments)
    def iterate(self):
        for conf in self.path: # [1:]
            set_joint_positions(self.robot, self.joints, conf)
            for attachment in self.attachments:
                attachment.assign()
            yield conf
    def __repr__(self):
        return 'm(#J{},#pt{},E{},{})'.format(len(self.joints), len(self.path),self.element,self.tag)
    def to_data(self):
        data = {}
        data['element'] = self.element
        data['tag'] = self.tag
        data['path'] = [list(conf) for conf in self.path]
        # data['attachments'] = [jsonpickle.encode(attach, keys=True) for attach in self.attachments]
        data['attachments'] = [attach.to_data() for attach in self.attachments]
        return data
    @classmethod
    def from_data(cls, data, robot, joints, attachments):
        path = data['path']
        element = data['element']
        tag = data['tag']
        return cls(robot, joints, path, attachments=attachments, element=element, tag=tag)

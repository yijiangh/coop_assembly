import argparse
import numpy as np
import pybullet as pb
# JOINT_REVOLUTE, JOINT_FIXED, JOINT_SPHERICAL

from pybullet_planning import joints_from_names, get_distance, get_joint_type, child_link_from_joint, oobb_from_points, \
    vertices_from_link, draw_oobb, wait_if_gui, draw_pose, get_link_name, get_joint_name, get_link_name, apply_alpha, RED, \
    vertices_from_rigid, set_color, get_aabb, draw_aabb, clone_body, remove_body, get_aabb_center, get_link_pose, get_movable_joints, \
    get_links

from coop_assembly.planning.utils import load_world
from coop_assembly.planning.robot_setup import EE_LINK_NAME, get_disabled_collisions, get_custom_limits, CONTROL_JOINT_NAMES, BASE_LINK_NAME, \
    TOOL_LINK_NAME, ROBOT_NAME, GANTRY_JOINT_LIMITS, IK_BASE_LINK_NAME, IK_JOINT_NAMES, ROBOT_URDF

#######################

def compute_joint_spheres(robot):
    for link in get_links(robot):
        set_color(robot, apply_alpha(RED, 0.1), link=link)
    jointspheres = {}
    # joints = joints_from_names(robot, CONTROL_JOINT_NAMES)
    joints = get_movable_joints(robot)

    for j in joints:
        if get_joint_type(robot, j) != pb.JOINT_REVOLUTE:
            continue

        # childlinks = j.GetHierarchyChildLink().GetRigidlyAttachedLinks()
        # for childlink in childlinks:
        # Tlink = childlink.GetTransform()
        # localaabb = childlink.ComputeLocalAABB()
        # linkpos = dot(Tlink[:3,:3], localaabb.pos()) + Tlink[:3,3]
        # extensiondist = linalg.norm(j.GetAnchor() - linkpos)
        # linkradius = linalg.norm(localaabb.extents())
        # sphereradius = max(sphereradius, linkradius+extensiondist)
        #############
        sphereradius = 0.0
        child_link = child_link_from_joint(j)
        print('Joint {} - Child {}'.format(get_joint_name(robot, j), get_link_name(robot, child_link)))
        link_pose = get_link_pose(robot, child_link)
        draw_pose(link_pose)

        # vertices = vertices_from_link(robot, child_link)
        # vertices = vertices_from_rigid(robot, child_link)
        # link_oobb = oobb_from_points(vertices)
        # draw_pose(link_oobb[1])
        # draw_oobb(link_oobb)
        # TODO use OOBB
        aabb = get_aabb(robot, child_link)
        draw_aabb(aabb)

        linkpos = get_aabb_center(aabb)
        wait_if_gui()


        # spherepos = j.GetAnchor()
        # # process any child joints
        # minpos = spherepos - sphereradius*ones([1,1,1])
        # maxpos = spherepos + sphereradius*ones([1,1,1])

        # childjoints = [testj for testj in self.robot.GetJoints() if testj.GetHierarchyParentLink() in childlinks]
        # for childjoint in childjoints:
        #     if childjoint.GetJointIndex() in jointspheres:
        #         childpos, childradius = jointspheres[childjoint.GetJointIndex()]
        #         minpos = numpy.minimum(minpos, childpos - childradius*ones([1,1,1]))
        #         maxpos = numpy.maximum(maxpos, childpos + childradius*ones([1,1,1]))

        # newspherepos = 0.5*(minpos + maxpos)
        # newsphereradius = linalg.norm(newspherepos - spherepos) + sphereradius
        # for childjoint in childjoints:
        #     if childjoint.GetJointIndex() in jointspheres:
        #         childpos, childradius = jointspheres[childjoint.GetJointIndex()]
        #         newsphereradius = max(newsphereradius, linalg.norm(newspherepos - childpos) + childradius)

        # # go through all the spheres and get the max radius
        # jointspheres[j.GetJointIndex()] = (numpy.around(newspherepos, 8), numpy.around(newsphereradius, 8))

    return jointspheres

def compute_joint_stats(xyzdelta=0.01, viewer=False):
    # https://github.com/rdiankov/openrave/blob/master/python/databases/linkstatistics.py#L128
    _, robot = load_world(use_floor=False, viewer=viewer)
    robot_cloned = robot
    # robot_cloned = clone_body(robot, visual=False, collision=True)
    # remove_body(robot)
    joint_spheres = compute_joint_spheres(robot_cloned)
    # return weights, resolutions

#######################

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    compute_joint_stats(viewer=args.viewer)

if __name__ == '__main__':
    main()

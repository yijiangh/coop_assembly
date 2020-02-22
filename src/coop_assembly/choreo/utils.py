from pybullet_planning import HideOutput, load_pybullet, set_static, set_joint_positions, joints_from_names, \
    create_plane, set_point, Point

from coop_assembly.choreo.robot_setup import get_picknplace_robot_data, get_robot_init_conf

def load_world(use_floor=True):
    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, _, _, joint_names, _ = robot_data

    print(robot_urdf)
    obstacles = []
    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
        # 1/velocity = weight
        # print([get_max_velocity(robot, joint) for joint in get_movable_joints(robot)])
        set_static(robot)
        set_joint_positions(robot, joints_from_names(robot, joint_names), get_robot_init_conf())
        if use_floor:
            floor = create_plane()
            obstacles.append(floor)
            set_point(floor, Point(x=1.2, z=0.023-0.025)) # -0.02
        else:
            floor = None
    return obstacles, robot

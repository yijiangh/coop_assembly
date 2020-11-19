using PyCall
PyPb = pyimport("pybullet_planning")

#########################################

"""
create a cylindrial element in the pybullet simulator
"""
function create_element(p1, p2, radius; color=PyPb.apply_alpha(PyPb.RED, alpha=1))
    height = norm(p2 - p1)
    center = (p1 + p2) / 2
    # extents = (p2 - p1) / 2
    delta = p2 - p1
    x, y, z = delta
    phi = atan(y, x)
    theta = acos(z / norm(delta))
    quat = PyPb.quat_from_euler(PyPb.Euler(pitch=theta, yaw=phi))
    # p1 is z=-height/2, p2 is z=+height/2

    # Visually, smallest diameter is 2e-3
    # TODO: boxes
    #body = create_cylinder(radius, height, color=color, mass=STATIC_MASS)
    body = PyPb.create_capsule(radius, height, color=color, mass=PyPb.STATIC_MASS)
    PyPb.set_point(body, center)
    PyPb.set_quat(body, quat)
    return body
end

function center_viewer(nodes, pitch=-π/8, distance=2)
    # TODO: be robust to SCALE
    centroid = mean([nodes[node]["point"] for node in keys(nodes)]; dims=1)[1]
    centroid[2] = min([nodes[node]["point"][3] for node in keys(nodes)]...)
    PyPb.set_camera(yaw=deg2rad(0), pitch=deg2rad(pitch), distance=distance, target_position=centroid)
    return PyPb.draw_pose(PyPb.Pose(point=centroid), length=1)
end


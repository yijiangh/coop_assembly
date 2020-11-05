# convert a bar network
function from_bar_network(data, debug=False)
    if "data" in keys(data)
        data = data["data"]
    end
    bar_network = PyCompasDataStructures.Network.from_data(data)

    b_struct = PyCoopDataStructures.BarStructure()
    centroid_pt = np.zeros(3)

    index_from_element = {}
    for (v, v_data) in bar_network.nodes(data=True)
        axis_endpts = v_data["axis_endpoints"]
        is_grounded = v_data["fixed"]
        radius = v_data["radius"]
        bar_key = b_struct.add_bar(v, [p for p in axis_endpts], "tube", nothing, nothing, radius=radius, grounded=is_grounded)
        index_from_element[v] = bar_key
    end

    for e in bar_network.edges()
        b_struct.connect_bars(index_from_element[e[0]], index_from_element[e[1]])
        contact_pts = compute_contact_line_between_bars(b_struct, index_from_element[e[0]], index_from_element[e[1]])
        b_struct.edge[index_from_element[e[0]]][index_from_element[e[1]]]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})
    end

    # * add grounded connector
    b_struct.generate_grounded_connection()

    element_bodies = b_struct.get_element_bodies(color=apply_alpha(RED, 0.5))
    if debug
        wait_if_gui("Parsed bar assembly.")
    end
    return b_struct
end

##################################################
# drawing and viz stuff

def draw_state(viewer, state, colors):
    viewer.clear()
    viewer.draw_environment()
    viewer.draw_robot(*state.conf[::-1])
    for block, pose in state.block_poses.items():
        r, c = pose[::-1]
        viewer.draw_block(r, c, name=block, color=colors[block])
    if state.holding is not None:
        pose = state.conf - GRASP
        r, c = pose[::-1]
        viewer.draw_block(r, c, name=state.holding, color=colors[state.holding])


def apply_action(state, action):
    conf, holding, block_poses = state
    # TODO: don't mutate block_poses?
    name, args = action
    if name == 'move':
        _, conf = args
    elif name == 'pick':
        holding, _, _ = args
        del block_poses[holding]
    elif name == 'place':
        block, pose, _ = args
        holding = None
        block_poses[block] = pose
    elif name == 'push':
        block, _, _, pose, conf = args
        holding = None
        block_poses[block] = pose
    else:
        raise ValueError(name)
    return DiscreteTAMPState(conf, holding, block_poses)

def apply_plan(tamp_problem, plan):
    colors = dict(zip(tamp_problem.initial.block_poses, COLORS))
    viewer = DiscreteTAMPViewer(1, len(tamp_problem.poses), title='Initial')
    state = tamp_problem.initial
    print(state)
    draw_state(viewer, state, colors)
    for action in plan:
        user_input('Continue?')
        state = apply_action(state, action)
        print(state)
        draw_state(viewer, state, colors)
    user_input('Finish?')

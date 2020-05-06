from pybullet_planning import connect, wait_if_gui, dump_world, create_plane, create_box

def test_connect(viewer=False):
    connect(use_gui=viewer)
    create_plane(color=(0.8, 0.8, 0.8))
    create_box(1,1,1)
    dump_world()
    wait_if_gui()

if __name__ == '__main__':
    test_connect(True)

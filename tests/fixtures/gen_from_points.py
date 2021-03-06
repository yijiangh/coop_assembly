import pytest

@pytest.fixture
def points_library():
    # test_set_name : (all_points, base_triangle_points)
    point_lib = {
    'single_tet' :(
                   [[585.71531, 100.962035, -0.009416],
                    [677.56771, -18.677707, -0.009416],
                    [524.632299, -36.037475, -0.009416],
                    [578.246553, 20.207032, 83.669489]],
                   [[585.71531, 100.962035, -0.009416],
                    [677.56771, -18.677707, -0.009416],
                    [524.632299, -36.037475, -0.009416]]
                   ),
    'single_cube' : ([[52, 0, 0],
                      [52, 52, 0],
                      [52, 52, 52],
                      [0, 52, 52],
                      [0, 52, 0],
                      [0, 0, 0],
                      [52, 0, 52],
                      [0, 0, 52]],
                     [[52, 0, 0],
                      [0, 52, 0],
                      [0, 0, 0]]),
    '12_bars' : ([[391.47375, -36.037475, -3.651681],
                    [544.409161, -18.677707, -3.651681],
                    [452.556761, 100.962035, -3.651681],
                    [445.002397, -26.403189, 65.681852],
                    [445.088004, 61.579393, 80.027223],
                    [503.035648, 43.28816, 57.058508]],
                    [[391.47375, -36.037475, -3.651681],
                     [544.409161, -18.677707, -3.651681],
                     [452.556761, 100.962035, -3.651681]]),
    }
    return point_lib

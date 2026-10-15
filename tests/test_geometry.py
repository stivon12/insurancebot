import math

import numpy as np

from pipeline.geometry import (
    Camera,
    fit_room_polygon,
    floor_point,
    integrate_yaw,
    oriented_extent,
    polygon_area,
    polygon_perimeter,
)


def cast(poly: np.ndarray, origin: np.ndarray, az: float) -> np.ndarray:
    """Nearest intersection of a ray from origin (azimuth clockwise from +y) with polygon edges."""
    d = np.array([math.sin(az), math.cos(az)])
    best = None
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        t, u = np.linalg.solve(m, a - origin)
        if t > 0 and 0 <= u <= 1 and (best is None or t < best):
            best = t
    return origin + best * d


def boundary(poly, origin, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pts = [cast(poly, origin, math.radians(a)) for a in np.arange(0, 360, 1.0)]
    pts = np.array(pts) - origin
    return pts * (1 + rng.normal(0, noise, (len(pts), 1)))


def test_rectangle_area_and_extent():
    room = np.array([[0, 0], [4.2, 0], [4.2, 3.1], [0, 3.1]])
    pts = boundary(room, np.array([1.5, 1.2]))
    poly = fit_room_polygon(pts)
    assert abs(polygon_area(poly) / (4.2 * 3.1) - 1) < 0.03
    length, width = oriented_extent(poly)
    assert abs(length - 4.2) < 0.1 and abs(width - 3.1) < 0.1
    assert abs(polygon_perimeter(poly) - 2 * (4.2 + 3.1)) < 0.3


def test_rotated_rectangle():
    room = np.array([[0, 0], [4.0, 0], [4.0, 3.0], [0, 3.0]])
    th = math.radians(27)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pts = boundary(room @ R.T, np.array([1.8, 1.4]) @ R.T)
    assert abs(polygon_area(fit_room_polygon(pts)) / 12.0 - 1) < 0.03


def test_l_shaped_room():
    room = np.array([[0, 0], [5, 0], [5, 2], [3, 2], [3, 4], [0, 4]])
    true_area = 5 * 2 + 3 * 2
    pts = boundary(room, np.array([1.5, 1.5]), noise=0.005)
    poly = fit_room_polygon(pts)
    assert len(poly) == 6
    assert abs(polygon_area(poly) / true_area - 1) < 0.05


def test_camera_floor_point_portrait_level():
    # Phone upright in portrait, level: up = device +y. Pixel 300 px below centre with f=1000.
    cam = Camera(f=1000, cx=540, cy=960, up=np.array([0.0, 1.0, 0.0]), azimuth=0.0)
    p = floor_point(cam, 540, 960 + 300, cam_height=1.4)
    expected = 1.4 / math.tan(math.atan(300 / 1000))
    assert abs(p[1] - expected) < 1e-6 and abs(p[0]) < 1e-6
    assert floor_point(cam, 540, 900, 1.4) is None  # above the horizon


def test_camera_pitched_down_and_azimuth():
    tilt = math.radians(20)  # camera pitched 20 deg down: the back of the phone faces partly downward,
    up = np.array([0.0, math.cos(tilt), math.sin(tilt)])  # so "up" gains a +z (screen-side) component
    cam = Camera(f=1000, cx=540, cy=960, up=up, azimuth=math.radians(90))
    elev, az = cam.elevation_azimuth(540, 960)
    assert abs(elev + tilt) < 1e-6 and abs(az - math.radians(90)) < 1e-6
    _, az_right = cam.elevation_azimuth(1000, 960)
    assert az_right > math.radians(90)  # pixels to the right are clockwise


def test_landscape_orientation():
    # Screen rotated 90 (top of phone to the left): the device's left edge (-x) points up.
    cam = Camera(f=1000, cx=960, cy=540, up=np.array([-1.0, 0.0, 0.0]), azimuth=0.0, orient=90)
    elev, _ = cam.elevation_azimuth(960, 540 + 200)
    assert elev < 0  # lower in the image is below the horizon


def test_integrate_yaw_quarter_turn_clockwise():
    # Portrait phone (up = +y) turning right at 90 deg/s for 1 s: rotation about up is negative (clockwise).
    imu = [{"t_ms": t, "ax": 0, "ay": 9.81, "az": 0, "ra": 0, "rb": 0, "rg": -90} for t in range(0, 1001, 10)]
    _, yaw = integrate_yaw(imu)
    assert abs(math.degrees(yaw[-1]) - 90) < 2

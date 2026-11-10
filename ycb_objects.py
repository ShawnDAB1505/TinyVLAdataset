"""
ycb_objects.py
Robust physical definitions for YCB-like realistic objects in PyBullet:
- Cup (Red Mug)
- Bowl (Blue Bowl)
- Plate (White Plate)
- Cracked Box (Yellow Cereal / Cracker Box)
- Apple (Red Fruit)
- Banana (Yellow Fruit)

Uses stable collision primitives (Cylinders, Boxes, Spheres) aligned with
visual OBJ meshes to guarantee 100% physically stable, upright standing
without any tipping, rolling, or tilting!
"""

import os
import math
import pybullet as p

ROBOSUITE_DIR = os.path.dirname(os.path.abspath(__file__))
_SHAPE_CACHE = {}

def spawn_cup(pos, scale=0.1, rgba=[0.9, 0.2, 0.2, 1.0], mass=0.15):
    """Spawns an upright Red Cup/Mug with flat bottom (physically stable)."""
    key = ("cup", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.033, height=0.068)
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "Cup.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, -0.034],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    p.changeDynamics(mb, -1, lateralFriction=2.0, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_bowl(pos, scale=0.09, rgba=[0.2, 0.4, 0.85, 1.0], mass=0.2):
    """Spawns an upright hollow Bowl with a graspable rim (+Z mouth)."""
    key = ("bowl", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        shape_types = [p.GEOM_CYLINDER]
        radii = [0.052]
        heights = [0.035]
        half_extents = [[0, 0, 0]]
        positions = [[0, 0, -0.015]]
        orientations = [[0, 0, 0, 1]]

        n_seg = 16
        r_ring = 0.069
        wall_half_thick = 0.0055
        seg_half_len = r_ring * math.tan(math.pi / n_seg) * 1.05
        for i in range(n_seg):
            ang = 2.0 * math.pi * i / n_seg
            shape_types.append(p.GEOM_BOX)
            radii.append(0)
            heights.append(0)
            half_extents.append([wall_half_thick, seg_half_len, 0.0325])
            positions.append([r_ring * math.cos(ang), r_ring * math.sin(ang), 0.0])
            orientations.append(p.getQuaternionFromEuler([0, 0, ang]))

        col = p.createCollisionShapeArray(
            shapeTypes=shape_types,
            radii=radii,
            lengths=heights,
            halfExtents=half_extents,
            collisionFramePositions=positions,
            collisionFrameOrientations=orientations
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "Bowl.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, -0.0325],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    p.changeDynamics(mb, -1, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_plate(pos, scale=0.08, rgba=[0.92, 0.92, 0.92, 1.0], mass=0.15):
    """Spawns a flat horizontal Plate with a raised outer rim and concave annular recess."""
    key = ("plate", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        shape_types = [p.GEOM_CYLINDER, p.GEOM_CYLINDER]
        radii = [0.092, 0.054]
        heights = [0.007, 0.020]
        half_extents = [[0, 0, 0], [0, 0, 0]]
        positions = [[0, 0, -0.0065], [0, 0, 0.0]]
        orientations = [[0, 0, 0, 1], [0, 0, 0, 1]]

        n_seg = 16
        r_ring = 0.095
        wall_half_thick = 0.0055
        seg_half_len = r_ring * math.tan(math.pi / n_seg) * 1.05
        for i in range(n_seg):
            ang = 2.0 * math.pi * i / n_seg
            shape_types.append(p.GEOM_BOX)
            radii.append(0)
            heights.append(0)
            half_extents.append([wall_half_thick, seg_half_len, 0.010])
            positions.append([r_ring * math.cos(ang), r_ring * math.sin(ang), 0.0])
            orientations.append(p.getQuaternionFromEuler([0, 0, ang]))

        col = p.createCollisionShapeArray(
            shapeTypes=shape_types,
            radii=radii,
            lengths=heights,
            halfExtents=half_extents,
            collisionFramePositions=positions,
            collisionFrameOrientations=orientations
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "plate.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, -0.01],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    p.changeDynamics(mb, -1, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_cracked_box(pos, orn=None, scale=0.060, rgba=[0.85, 0.65, 0.15, 1.0], mass=0.25):
    """Spawns a Cereal/Cracker Box (supports custom orientation for leaning/lying down)."""
    key = ("cracked_box", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        s_ratio = float(scale) / 0.075
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.045 * s_ratio, 0.045 * s_ratio, 0.12 * s_ratio])
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "cracked_box.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[-0.02 * s_ratio, 0, 0],
            visualFrameOrientation=p.getQuaternionFromEuler([0, -math.pi / 2, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=[0, 0, 0, 1] if orn is None else orn
    )
    p.changeDynamics(mb, -1, lateralFriction=3.0, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_apple(pos, scale=0.045, rgba=[0.85, 0.15, 0.15, 1.0], mass=0.1):
    """Spawns an Apple (natural sphere-like fruit)."""
    key = ("apple", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.035)
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "apple.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, -0.035],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    p.changeDynamics(mb, -1, lateralFriction=1.5, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_banana(pos, scale=0.045, rgba=[0.95, 0.85, 0.1, 1.0], mass=0.08):
    """Spawns a Banana resting naturally on the table."""
    key = ("banana", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.02, height=0.12,
                                     collisionFrameOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "Banana.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, 0],
            visualFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                           basePosition=pos, baseOrientation=[0, 0, 0, 1])
    p.changeDynamics(mb, -1, lateralFriction=2.0, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_scissors(pos, orn=None, scale=0.08, rgba=[0.12, 0.35, 0.88, 1.0], mass=0.10):
    """Spawns a pair of Scissors lying flat horizontally on the table."""
    key = ("scissors", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.050, 0.005])
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "scissors.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, -0.036 * scale],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=[0, 0, 0, 1] if orn is None else orn
    )
    p.changeDynamics(mb, -1, lateralFriction=3.0, spinningFriction=0.2, rollingFriction=0.05)
    return mb

def spawn_teapot(pos, scale=0.075, rgba=[0.88, 0.75, 0.18, 1.0], mass=0.40):
    """Spawns a large upright Teapot (household ceramic tea pot) with flat stable base."""
    key = ("teapot", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.055, height=0.120)
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "teapot.obj"),
            meshScale=[scale]*3,
            visualFramePosition=[0, 0, -0.060],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
    p.changeDynamics(mb, -1, lateralFriction=3.0, spinningFriction=0.2, rollingFriction=0.05)
    return mb


def spawn_cardboard(pos, orn=None, scale=1.0, rgba=[0.76, 0.56, 0.34, 1.0], mass=0.18):
    """
    Spawns an UPSIDE-DOWN (úp xuống) hollow Cardboard Box (cardboard.obj).
    The interior cavity (7.2cm x 11.0cm x 4.8cm) is completely hollow from underneath
    so that Scissors (5.0cm x 10.0cm x 1.0cm) can lie hidden underneath without collision,
    while the top body (width 8.0cm) can be grasped cleanly by the 2-finger gripper.
    """
    key = ("cardboard_upside_down", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        s_types = [p.GEOM_BOX] * 5
        h_exts = [
            [0.040 * scale, 0.060 * scale, 0.004 * scale],  # Top closed base (facing up when upside-down)
            [0.003 * scale, 0.060 * scale, 0.027 * scale],  # -X side wall
            [0.003 * scale, 0.060 * scale, 0.027 * scale],  # +X side wall
            [0.040 * scale, 0.003 * scale, 0.027 * scale],  # -Y front wall
            [0.040 * scale, 0.003 * scale, 0.027 * scale],  # +Y back wall
        ]
        c_pos = [
            [0.0, 0.0, 0.023 * scale],
            [-0.038 * scale, 0.0, 0.0],
            [+0.038 * scale, 0.0, 0.0],
            [0.0, -0.058 * scale, 0.0],
            [0.0, +0.058 * scale, 0.0],
        ]
        c_orn = [[0, 0, 0, 1]] * 5
        col = p.createCollisionShapeArray(
            shapeTypes=s_types,
            halfExtents=h_exts,
            collisionFramePositions=c_pos,
            collisionFrameOrientations=c_orn
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "cardboard.obj"),
            meshScale=[0.120 * scale, 0.180 * scale, 0.205 * scale],
            visualFramePosition=[0, 0, 0.027 * scale],
            visualFrameOrientation=p.getQuaternionFromEuler([-math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=[0, 0, 0, 1] if orn is None else orn
    )
    p.changeDynamics(mb, -1, lateralFriction=3.0, spinningFriction=0.2, rollingFriction=0.05)
    return mb


def spawn_book(pos, orn=None, scale=0.075, rgba=[0.18, 0.55, 0.45, 1.0], mass=0.20):
    """Spawns a Book (book.obj) lying flat on the table."""
    key = ("book", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.038, 0.053, 0.008])
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "book.obj"),
            meshScale=[scale] * 3,
            visualFramePosition=[0, 0, -0.005],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=[0, 0, 0, 1] if orn is None else orn
    )
    p.changeDynamics(mb, -1, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05)
    return mb


def spawn_car_toy(pos, orn=None, scale=0.10, rgba=[0.92, 0.45, 0.12, 1.0], mass=0.15):
    """Spawns a Toy Car (car_toy.obj) resting stably on the table."""
    key = ("car_toy", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.020, 0.049, 0.015])
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "car_toy.obj"),
            meshScale=[scale] * 3,
            visualFramePosition=[0, 0, -0.014],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=[0, 0, 0, 1] if orn is None else orn
    )
    p.changeDynamics(mb, -1, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05)
    return mb


def spawn_iphone(pos, orn=None, scale=0.11, rgba=[0.22, 0.25, 0.30, 1.0], mass=0.12):
    """Spawns an iPhone smartphone (iphone.obj) lying flat on the table."""
    key = ("iphone", float(scale), tuple(rgba))
    if key not in _SHAPE_CACHE:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.047, 0.024, 0.008])
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=os.path.join(ROBOSUITE_DIR, "iphone.obj"),
            meshScale=[scale] * 3,
            visualFramePosition=[0, 0, -0.008],
            visualFrameOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
            rgbaColor=rgba
        )
        _SHAPE_CACHE[key] = (col, vis)
    col, vis = _SHAPE_CACHE[key]
    mb = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=[0, 0, 0, 1] if orn is None else orn
    )
    p.changeDynamics(mb, -1, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05)
    return mb


def compute_affordance_grasp(obj_type, obj_id, preferred_angle_deg=None):
    """
    Universal Household Object Affordance Grasp Planner (dung chung cho moi Scenario & moi Object).
    Tu dong tinh toan diem dat tool0 (da bu chieu dai ngon kep TCP = 0.160m), goc xoay (orn),
    va do dong/mo ngon kep (grip_open, grip_close) phu hop voi hinh hoc vat ly cua tung vat the:
      - 'bowl'        : Kep vanh bat (Rim Grasp) tai r = 0.069m (1 ngon trong long, 1 ngon ngoai vanh)
      - 'plate'       : Kep mep dia (Edge Pinch Grasp) tai r = 0.095m (1 ngon trong ranh, 1 ngon ngoai mep)
      - 'cup'         : Kep than tren / mieng coc (Upper Wall Grasp, cach mieng coc >3cm)
      - 'apple'       : Kep xich dao qua tao (Spherical Equatorial Grasp)
      - 'banana'      : Kep ngang lung qua chuoi vuong goc truc dai (Cross-Axis Waist Grasp)
      - 'cracked_box' : Kep 2 mat ben hep cua hop (Parallel Box Face Grasp)
    """
    pos, orn = p.getBasePositionAndOrientation(obj_id)
    _, _, obj_yaw = p.getEulerFromQuaternion(orn)
    kind = str(obj_type).strip().lower()

    if kind == "bowl":
        ang_deg = 90.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        ang_rad = math.radians(ang_deg)
        r_rim = 0.069
        target_pos = [pos[0] + r_rim * math.cos(ang_rad), pos[1] + r_rim * math.sin(ang_rad), pos[2] + 0.175]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, ang_rad])
        return {
            "affordance_type": "bowl_rim_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": ang_deg,
            "grip_open": 0.055,
            "grip_close": 0.00
        }

    if kind == "plate":
        ang_deg = 135.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        ang_rad = math.radians(ang_deg)
        r_edge = 0.095
        # Yaw cua ngon kep vuong goc voi tiep tuyen vien dia (radially aligned)
        yaw_deg = ((ang_deg + 180.0) % 360.0) - 360.0 if ang_deg >= 0 else ang_deg
        target_pos = [pos[0] + r_edge * math.cos(ang_rad), pos[1] + r_edge * math.sin(ang_rad), pos[2] + 0.155]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "plate_edge_pinch",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.055,
            "grip_close": 0.00
        }

    if kind == "banana":
        # Truc dai cua chuoi nam doc theo truc X cuc bo -> kep vuong goc voi truc dai (obj_yaw + 90°)
        yaw_deg = math.degrees(obj_yaw) + 90.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.154]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "banana_waist_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.065,
            "grip_close": 0.005
        }

    if kind == "apple":
        yaw_deg = 0.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.165]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "apple_equatorial_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.08,
            "grip_close": 0.018
        }

    if kind in ("cracked_box", "box"):
        yaw_deg = math.degrees(obj_yaw) if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.168]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "box_parallel_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.075,
            "grip_close": 0.020
        }

    if kind in ("scissors", "scissor"):
        yaw_deg = 90.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.159]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "scissors_flat_pinch",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.035,
            "grip_close": 0.000
        }

    if kind in ("teapot", "pot"):
        yaw_deg = 0.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.190]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "teapot_top_lid_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.085,
            "grip_close": 0.035
        }

    if kind in ("cardboard", "cardbox"):
        yaw_deg = math.degrees(obj_yaw) if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.168]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "cardboard_top_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.095,
            "grip_close": 0.025
        }

    if kind == "book":
        yaw_deg = math.degrees(obj_yaw) if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.150]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "book_spine_grasp",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.090,
            "grip_close": 0.000
        }

    if kind in ("car_toy", "car"):
        yaw_deg = math.degrees(obj_yaw) if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.154]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "car_body_pinch",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.075,
            "grip_close": 0.008
        }

    if kind in ("iphone", "phone"):
        yaw_deg = math.degrees(obj_yaw) + 90.0 if preferred_angle_deg is None else float(preferred_angle_deg)
        target_pos = [pos[0], pos[1], pos[2] + 0.152]
        target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
        return {
            "affordance_type": "iphone_flat_pinch",
            "target_pos": target_pos,
            "target_orn": target_orn,
            "yaw_deg": yaw_deg,
            "grip_open": 0.075,
            "grip_close": 0.008
        }

    # Mặc định: 'cup' (Kẹp thân trên / miệng cốc)
    yaw_deg = 0.0 if preferred_angle_deg is None else float(preferred_angle_deg)
    target_pos = [pos[0], pos[1], pos[2] + 0.182]
    target_orn = p.getQuaternionFromEuler([math.pi, 0, math.radians(yaw_deg)])
    return {
        "affordance_type": "cup_upper_wall_grasp",
        "target_pos": target_pos,
        "target_orn": target_orn,
        "yaw_deg": yaw_deg,
        "grip_open": 0.08,
        "grip_close": 0.020
    }


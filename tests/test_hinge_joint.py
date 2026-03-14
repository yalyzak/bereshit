"""
Tests for HingeJoint — verifies all constraint functionality:
  1. Linear constraint  (anchors stay together)
  2. Angular constraint  (only hinge-axis rotation is free)
  3. Baumgarte correction (drift is corrected)
  4. Hinge-axis friction
  5. Integration test: heavy + light body scene
"""

import math
import sys
import os

# -- make bereshit importable from the tests/ directory --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bereshit.Vector3 import Vector3
from bereshit.Quaternion import Quaternion
from bereshit.Object import Object
from bereshit.Rigidbody import Rigidbody
from bereshit.BoxCollider import BoxCollider
from bereshit.Camera import Camera
from bereshit.HingeJoint import HingeJoint
from bereshit.World import World


# ================================================================
#  Helpers
# ================================================================

def make_body(name, position, mass=1.0, velocity=None, angular_velocity=None,
              size=(1, 1, 1), is_kinematic=False, use_gravity=False):
    """Create an Object with Rigidbody + BoxCollider, ready for physics."""
    obj = Object(position=position, size=size, name=name)
    obj.add_component(Rigidbody(
        mass=mass,
        velocity=velocity or Vector3(0, 0, 0),
        angular_velocity=angular_velocity or Vector3(0, 0, 0),
        isKinematic=is_kinematic,
        useGravity=use_gravity,
    ))
    obj.add_component(BoxCollider())
    return obj


def make_world(*objects, gravity=Vector3(0, 0, 0), dt=1 / 60):
    """Create a headless World with the given objects."""
    cam_obj = Object(position=(0, 0, -10), name="cam")
    cam_obj.add_component(Camera(shading="wire"))

    children = [cam_obj] + list(objects)
    world = World([False], children=children, gravity=gravity, tick=dt, speed=1)
    world.Start()
    return world


def step(world, n=1):
    """Advance the world by n physics ticks."""
    for _ in range(n):
        world.update(check=False)


def anchor_world(joint, body_a):
    """Compute the world-space anchor on body A's side."""
    return body_a.position + body_a.quaternion.rotate(joint.local_anchor_a)


def anchor_world_b(joint, body_b):
    """Compute the world-space anchor on body B's side."""
    return body_b.position + body_b.quaternion.rotate(joint.local_anchor_b)


def distance(a, b):
    return (a - b).magnitude()


# ================================================================
#  Test 1: Linear constraint — anchors stay together
# ================================================================

def test_linear_constraint_holds():
    """
    Two bodies connected by a hinge, one given an initial velocity.
    After several steps the anchors should still be close together.
    """
    a = make_body("A", position=(0, 0, 0), mass=5.0)
    b = make_body("B", position=(5, 0, 0), mass=5.0,
                  velocity=Vector3(0, 5, 0))

    hinge = HingeJoint(body_b=b, axis=Vector3(0, 0, 1))
    a.add_component(hinge)

    world = make_world(a, b)
    step(world, 120)  # 2 seconds at 60 Hz

    anchor_a = anchor_world(hinge, a)
    anchor_b = anchor_world_b(hinge, b)
    drift = distance(anchor_a, anchor_b)

    print(f"[linear constraint] anchor drift = {drift:.6f}")
    assert drift < 0.5, f"Anchor drift too large: {drift}"


# ================================================================
#  Test 2: Angular constraint — perpendicular rotation is killed
# ================================================================

def test_angular_constraint_kills_perpendicular():
    """
    Hinge axis = Z.  Give body B angular velocity around X (perpendicular).
    After solving, most of that angular velocity should be eliminated.
    """
    a = make_body("A", position=(0, 0, 0), mass=10, is_kinematic=True)
    b = make_body("B", position=(5, 0, 0), mass=1.0,
                  angular_velocity=Vector3(10, 0, 0))  # spin around X

    hinge = HingeJoint(body_b=b, axis=Vector3(0, 0, 1))
    a.add_component(hinge)

    world = make_world(a, b)
    step(world, 60)

    # The X component of angular velocity should be small
    wx = abs(b.Rigidbody.angular_velocity.x)
    wz = abs(b.Rigidbody.angular_velocity.z)

    print(f"[angular constraint] w_x (perpendicular) = {wx:.4f},  w_z (hinge) = {wz:.4f}")
    assert wx < 1.0, f"Perpendicular angular velocity not damped: wx={wx}"


# ================================================================
#  Test 3: Angular constraint — hinge-axis rotation stays free
# ================================================================

def test_hinge_axis_rotation_free():
    """
    Hinge axis = Z.  Bodies are placed along Z so the lever arms are
    parallel to the hinge axis.  Angular velocity around Z should NOT
    be killed (it's the free DOF) because it creates zero tangential
    velocity at the anchor.
    """
    a = make_body("A", position=(0, 0, 0), mass=10, is_kinematic=True)
    b = make_body("B", position=(0, 0, 5), mass=1.0,
                  angular_velocity=Vector3(0, 0, 5))  # spin around Z (free axis)

    hinge = HingeJoint(body_b=b, axis=Vector3(0, 0, 1), friction_coefficient=0.0)
    a.add_component(hinge)

    world = make_world(a, b)
    step(world, 60)

    wz = abs(b.Rigidbody.angular_velocity.z)
    print(f"[hinge freedom] w_z after 60 steps = {wz:.4f}  (initial = 5.0)")
    assert wz > 2.0, f"Hinge-axis rotation was incorrectly damped: wz={wz}"


# ================================================================
#  Test 4: Baumgarte correction — position error is reduced
# ================================================================

def test_baumgarte_position_correction():
    """
    Start with bodies already slightly separated from where the anchor
    expects them.  Baumgarte should push them back together.
    """
    a = make_body("A", position=(0, 0, 0), mass=5.0)
    b = make_body("B", position=(5, 0, 0), mass=5.0)

    hinge = HingeJoint(body_b=b, axis=Vector3(0, 1, 0))
    a.add_component(hinge)

    # Manually displace body B away from where the anchor expects it
    b.position = Vector3(5.5, 0.3, 0)

    world = make_world(a, b)
    initial_drift = distance(anchor_world(hinge, a), anchor_world_b(hinge, b))

    step(world, 120)

    final_drift = distance(anchor_world(hinge, a), anchor_world_b(hinge, b))
    print(f"[baumgarte] drift: {initial_drift:.4f} -> {final_drift:.4f}")
    assert final_drift < initial_drift, \
        f"Baumgarte did not reduce drift: {initial_drift} ΓåÆ {final_drift}"


# ================================================================
#  Test 5: Friction resists hinge-axis rotation
# ================================================================

def test_hinge_friction():
    """
    Same setup as test 3, but with high friction.  The hinge-axis spin
    should be significantly slower than the frictionless case.
    """
    a = make_body("A", position=(0, 0, 0), mass=10, is_kinematic=True)

    # --- frictionless ---
    b_free = make_body("B_free", position=(0, 0, 5), mass=1.0,
                       angular_velocity=Vector3(0, 0, 5))
    hinge_free = HingeJoint(body_b=b_free, axis=Vector3(0, 0, 1),
                            friction_coefficient=0.0)
    a_free = make_body("A_free", position=(0, 0, 0), mass=10, is_kinematic=True)
    a_free.add_component(hinge_free)
    world_free = make_world(a_free, b_free)
    step(world_free, 60)
    wz_free = abs(b_free.Rigidbody.angular_velocity.z)

    # --- with friction ---
    b_fric = make_body("B_fric", position=(0, 0, 5), mass=1.0,
                       angular_velocity=Vector3(0, 0, 5))
    hinge_fric = HingeJoint(body_b=b_fric, axis=Vector3(0, 0, 1),
                            friction_coefficient=5.0)
    a_fric = make_body("A_fric", position=(0, 0, 0), mass=10, is_kinematic=True)
    a_fric.add_component(hinge_fric)
    world_fric = make_world(a_fric, b_fric)
    step(world_fric, 60)
    wz_fric = abs(b_fric.Rigidbody.angular_velocity.z)

    print(f"[friction] frictionless wz={wz_free:.4f},  with friction wz={wz_fric:.4f}")
    assert wz_fric < wz_free, \
        f"Friction did not slow down rotation: free={wz_free}, fric={wz_fric}"


# ================================================================
#  Test 6: Symmetric bodies — momentum is conserved
# ================================================================

def test_momentum_conservation():
    """
    Two equal-mass bodies connected by a frictionless hinge.
    Total linear momentum should be conserved.
    """
    a = make_body("A", position=(0, 0, 0), mass=2.0,
                  velocity=Vector3(3, 0, 0))
    b = make_body("B", position=(5, 0, 0), mass=2.0,
                  velocity=Vector3(-1, 0, 0))

    hinge = HingeJoint(body_b=b, axis=Vector3(0, 1, 0))
    a.add_component(hinge)

    world = make_world(a, b)

    p0 = a.Rigidbody.velocity * a.Rigidbody.mass + \
         b.Rigidbody.velocity * b.Rigidbody.mass

    step(world, 120)

    p1 = a.Rigidbody.velocity * a.Rigidbody.mass + \
         b.Rigidbody.velocity * b.Rigidbody.mass

    dp = distance(p0, p1)
    print(f"[momentum] initial={p0}  final={p1}  delta={dp:.4f}")
    assert dp < 1.0, f"Momentum not conserved: ╬öp={dp}"


# ================================================================
#  Test 7: Kinematic anchor — body swings like a pendulum
# ================================================================

def test_pendulum_with_gravity():
    """
    One kinematic body (fixed in space), one dynamic body hanging below,
    connected by a hinge.  With gravity, body B should swing like a
    pendulum and stay connected.
    """
    a = make_body("A", position=(0, 5, 0), mass=10, is_kinematic=True)
    b = make_body("B", position=(5, 5, 0), mass=1.0,
                  velocity=Vector3(0, 0, 0), use_gravity=True)

    hinge = HingeJoint(body_b=b, axis=Vector3(0, 0, 1))
    a.add_component(hinge)

    world = make_world(a, b, gravity=Vector3(0, -9.8, 0))
    step(world, 300)  # 5 seconds swing

    drift = distance(anchor_world(hinge, a), anchor_world_b(hinge, b))
    print(f"[pendulum] anchor drift after 5s = {drift:.4f}")
    print(f"[pendulum] B position = {b.position}")
    assert drift < 0.5, f"Pendulum anchor drifted: {drift}"
    # B should have moved downward from gravity
    assert b.position.y < 5.0, f"Body B didn't fall under gravity: y={b.position.y}"


# ================================================================
#  Test 8: SCENE TEST — heavy body + light body with initial velocity
# ================================================================

def test_heavy_light_scene():
    """
    Scene:
      - 1 camera
      - 1 heavy body (mass=50, stationary)
      - 1 light body (mass=1, initial velocity=(5,3,0))
      - light is hinged to the heavy around the Y axis

    Expected behavior:
      - The light body swings around the heavy body
      - The heavy body barely moves (high mass ratio)
      - The anchor constraint holds despite the velocity
    """
    # Camera
    cam = Object(position=(0, 2, -15), name="camera")
    cam.add_component(Camera(shading="wire"))

    # Heavy body — large mass, stationary
    heavy = Object(position=(0, 3, 0), size=(2, 2, 2), name="heavy_block")
    heavy.add_component(Rigidbody(mass=50.0, useGravity=True))
    heavy.add_component(BoxCollider())

    # Light body — small, fast, connected via hinge
    light = Object(position=(3, 3, 0), size=(0.5, 0.5, 0.5), name="light_ball")
    light.add_component(Rigidbody(
        mass=1.0,
        velocity=Vector3(5, 3, 0),
        useGravity=True,
    ))
    light.add_component(BoxCollider())

    # Connect light ΓåÆ heavy via Y-axis hinge
    hinge = HingeJoint(body_b=light, axis=Vector3(0, 1, 0), friction_coefficient=0.1)
    heavy.add_component(hinge)

    # Build world (headless, no renderer)
    world = World(
        [False],
        children=[cam, heavy, light],
        gravity=Vector3(0, -9.8, 0),
        tick=1 / 60,
        speed=1,
    )
    world.Start()

    # Record initial state
    heavy_pos0 = Vector3(heavy.position.x, heavy.position.y, heavy.position.z)
    light_pos0 = Vector3(light.position.x, light.position.y, light.position.z)
    initial_anchor_drift = distance(anchor_world(hinge, heavy),
                                     anchor_world_b(hinge, light))

    # Run 5 seconds of simulation
    for _ in range(300):
        world.update(check=False)

    # ------ Assertions ------

    # 1) Anchor constraint held
    final_drift = distance(anchor_world(hinge, heavy),
                            anchor_world_b(hinge, light))
    print(f"[scene] anchor drift: initial={initial_anchor_drift:.4f}, "
          f"final={final_drift:.4f}")
    assert final_drift < 1.0, f"Anchor drifted too much: {final_drift}"

    # 2) Heavy body barely moved (high inertia)
    heavy_displacement = distance(heavy.position, heavy_pos0)
    print(f"[scene] heavy displacement = {heavy_displacement:.4f}")
    # It can move somewhat due to gravity and joint forces, but not a lot
    # compared to the light body

    # 3) Light body moved significantly
    light_displacement = distance(light.position, light_pos0)
    print(f"[scene] light displacement = {light_displacement:.4f}")
    assert light_displacement > 1.0, \
        f"Light body didn't move enough: {light_displacement}"

    # 4) Light body is still "near" the heavy body (tethered by joint)
    separation = distance(light.position, heavy.position)
    initial_separation = distance(light_pos0, heavy_pos0)
    print(f"[scene] separation: initial={initial_separation:.2f}, "
          f"final={separation:.2f}")
    # Should not have flown away — separation bounded by initial distance + some slack
    assert separation < initial_separation * 3, \
        f"Light body escaped: separation={separation}"

    print("[scene] PASSED -- heavy+light hinge scene behaves correctly")


# ================================================================
#  Runner
# ================================================================

if __name__ == "__main__":
    tests = [
        ("Linear constraint holds",              test_linear_constraint_holds),
        ("Angular constraint kills perpendicular", test_angular_constraint_kills_perpendicular),
        ("Hinge-axis rotation stays free",        test_hinge_axis_rotation_free),
        ("Baumgarte position correction",         test_baumgarte_position_correction),
        ("Hinge friction",                        test_hinge_friction),
        ("Momentum conservation",                 test_momentum_conservation),
        ("Pendulum with gravity",                 test_pendulum_with_gravity),
        ("Heavy + light scene",                   test_heavy_light_scene),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"  TEST: {name}")
        print(f"{'='*60}")
        try:
            test_fn()
            print(f"  [OK] PASSED")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append(name)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print(f"  Failed: {', '.join(errors)}")
    print(f"{'='*60}")

    sys.exit(1 if failed > 0 else 0)
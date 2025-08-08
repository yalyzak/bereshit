import asyncio
import copy
import threading
import time


import render as render
from bereshit import Object
# import old_render as render


def run(scene,speed=1,gizmos=False):
    TARGET_FPS = 60
    # bereshit.dt = TARGET_FPS * 0.000165
    if gizmos:
        point = Object(position=(0, 0, 0), size=(.1, .1, .1))
        hit_point_gizmos = [Object(position=(0, 0, 0), size=(.1, .1, .1),children=[Object(position=(0, 0, 0), size=(.1, .1, .1)) for i in range(8)]) for i in scene.get_all_colliders()]
        gizmos_container = Object(position=(0, 0, 0), size=(0, 0, 0), children=hit_point_gizmos, name='gizmos_container')
        world = Object(position=(0, 0, 0), size=(0, 0, 0), children=[scene,gizmos_container], name='world')
    else:
        world = Object(position=(0, 0, 0), size=(0, 0, 0), children=[scene], name='world')
    dt = 1 / 60

    startg = time.time()
    FPS = 1

    async def main_logic():
        start_wall_time = time.time()
        steps = 0
        # speed = 1  # real time slip
        # bereshit.dt = (10 / ((1 / dt) / 60) * speed)
        world.Start()
        while True:
            steps += 1
            simulated_time = steps * dt

            if steps % 10 == 0:
                world.update(dt, chack=True,gizmos=gizmos)
            else:
                # Update simulation
                world.update(dt,gizmos=gizmos)
            # Compute when, in wall clock time, this simulated time should happen
            # For double speed: simulated_time advances twice as fast as real time
            target_wall_time = start_wall_time + (simulated_time / speed)
            now = time.time()
            sleep_time = target_wall_time - now

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def start_async_loop():
        asyncio.run(main_logic())

    # if __name__ == "__main__":
    # Initialize world
    # world.reset_to_default()
    # start_async_loop()
    # Start async logic in a thread
    logic_thread = threading.Thread(target=start_async_loop, daemon=True)
    logic_thread.start()

    # Start rendering in main thread
    render.run_renderer(world)

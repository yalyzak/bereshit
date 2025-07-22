import asyncio
import threading
import time

import bereshit
import render



def run(world,speed=1):
    TARGET_FPS = 60
    # bereshit.dt = TARGET_FPS * 0.000165

    dt = 1 / 600

    startg = time.time()
    FPS = 1

    async def main_logic():
        start_wall_time = time.time()
        steps = 0
        # speed = 5  # real time slip
        bereshit.dt = (10 / ((1 / dt) / 60) * speed)
        world.Start()
        while True:
            steps += 1
            simulated_time = steps * dt

            if steps % 10 == 0:
                world.update(dt, chack=True)
            else:
                # Update simulation
                world.update(dt)
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

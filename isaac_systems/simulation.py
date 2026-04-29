from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.sensor import Camera, LidarRtx
import numpy as np
import csv
import time

# =========================
# CONFIG (experiment setup)
# =========================
DISTANCES = [2, 5, 10]
NOISE_LEVELS = [0.0, 0.05, 0.1]
FRAMES_PER_SCENARIO = 5

# =========================
# SETUP WORLD
# =========================
world = World()
world.scene.add_default_ground_plane()

# Cube (target object)
cube = world.scene.add( world.scene._scene.add_default_cube( prim_path="/World/Cube", position=np.array([0, 0, 1])) )

# Camera
camera = Camera( prim_path="/World/Camera", position=np.array([2, 2, 2]), frequency=10 )
camera.initialize()

# LiDAR
lidar = LidarRtx( prim_path="/World/Lidar", position=np.array([2, 0, 1]) )
lidar.initialize()

# =========================
# METRICS FUNCTIONS
# =========================
def compute_metrics(points, distance):
    num_points = len(points)

    if num_points > 0:
        depths = np.linalg.norm(points, axis=1)
        mean_depth = np.mean(depths)
        var_depth = np.var(depths)
    else:
        mean_depth = 0
        var_depth = 0

    density = num_points / (distance + 1e-5)

    return {
        "num_points": num_points,
        "mean_depth": mean_depth,
        "var_depth": var_depth,
        "density": density
    }

def add_noise(points, noise_level):
    if len(points) == 0:
        return points
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise

# =========================
# RUN EXPERIMENTS
# =========================
results = []

for dist in DISTANCES:
    for noise in NOISE_LEVELS:

        # Move cube
        cube.set_world_pose(position=np.array([dist, 0, 1]))

        # Warmup frames
        for _ in range(3):
            world.step(render=True)

        frame_times = []

        for frame in range(FRAMES_PER_SCENARIO):

            start = time.time()
            world.step(render=True)
            end = time.time()

            frame_time = (end - start) * 1000  # ms
            frame_times.append(frame_time)

            # Get LiDAR data
            points = lidar.get_point_cloud()

            # Add synthetic noise
            points = add_noise(points, noise)

            metrics = compute_metrics(points, dist)

            result = {
                "distance": dist,
                "noise": noise,
                "frame": frame,
                "frame_time_ms": frame_time,
                **metrics
            }

            results.append(result)

            print(
                f"[dist={dist}, noise={noise}] "
                f"points={metrics['num_points']} "
                f"depth={metrics['mean_depth']:.2f} "
                f"time={frame_time:.2f}ms"
            )

# =========================
# SAVE RESULTS
# =========================
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "distance", "noise", "frame",
            "frame_time_ms",
            "num_points", "mean_depth",
            "var_depth", "density"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

simulation_app.close()

print("\nSaved to results.csv")
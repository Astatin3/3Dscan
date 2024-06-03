from freenect2 import Device, FrameType
import psutil

from PIL import Image
import open3d as o3d
import numpy as np

def depth_to_points(depth_map):
    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    points = np.stack((x_coords, y_coords, depth_map), axis=-1)
    points = points.reshape(-1, 3)

    return points

def remove_invalid_points(points):
    mask = points[:, 2] != 0
    points = points[mask]

    return points, mask


def undistort_points(points, params):
    fx = params.fx
    fy = params.fy
    cx = params.cx
    cy = params.cy
    k1 = params.k1
    k2 = params.k2
    k3 = params.k3
    p1 = params.p1
    p2 = params.p2

    # Normalize points
    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy

    # Compute radial distances
    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3

    # Compute radial distortion
    radial_distortion = (1 + k1 * r2 + k2 * r4 + k3 * r6)

    # Compute tangential distortion
    xy = x * y
    xy2 = 2 * xy
    x2 = x ** 2
    y2 = y ** 2
    tangential_distortion_x = p1 * xy2 + p2 * (r2 + 2 * x2)
    tangential_distortion_y = p1 * (r2 + 2 * y2) + p2 * xy2

    # Undistort points
    undistorted_x = (x - tangential_distortion_x) / radial_distortion
    undistorted_y = (y - tangential_distortion_y) / radial_distortion

    # Reproject points
    undistorted_points = np.zeros_like(points)
    undistorted_points[:, 0] = undistorted_x * fx + cx
    undistorted_points[:, 1] = undistorted_y * fy + cy
    undistorted_points[:, 2] = points[:, 2]  # Preserve depth values

    return undistorted_points


voxel_size = 0.005

vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

reconstruction = o3d.geometry.PointCloud()
reconstruction.points = o3d.utility.Vector3dVector(np.random.random((5, 3)))
vis.add_geometry(reconstruction)

import numpy as np

threads = []


running = True
device = Device()

color_frame = None

n_cpus = psutil.cpu_count()


with device.running():

    for type_, frame in device:

        if type_ == FrameType.Color:
            img = frame.to_image()
            img = img.resize((424,512))
            color_frame = np.array(img)
            color_frame = color_frame[:, :, :3]
            width, height = color_frame.shape[:2]
            color_frame = color_frame.reshape(width * height, 3)

        elif type_ == FrameType.Depth and \
            color_frame is not None:

            depth_map = frame.to_array()
            depth_map /= 5
            # ir_image /= ir_image.max()
            # ir_image = np.sqrt(ir_image)


            points = depth_to_points(depth_map)
            points, mask = remove_invalid_points(points)
            points = undistort_points(points, device.ir_camera_params)

            points /= 512

            reconstruction.points = o3d.utility.Vector3dVector(points)
            # reconstruction.colors = o3d.utility.Vector3dVector(color_frame[mask])
            vis.update_geometry(reconstruction)

        if not vis.poll_events():
            break

        vis.update_renderer()

vis.close()

running = False

print("joining threads...")

for thread in threads:
    thread.join()
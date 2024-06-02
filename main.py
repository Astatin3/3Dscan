from freenect2 import Device, FrameType
import psutil
import cv2

import open3d as o3d
import numpy as np

from threading import Thread
from copy import deepcopy

voxel_size = 0.005

vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

reconstruction = o3d.geometry.PointCloud()
reconstruction.points = o3d.utility.Vector3dVector(np.random.random((5, 3)))
vis.add_geometry(reconstruction)


threads = []

def parse_frame(depth_frame, color_frame):

    global reconstruction


    undistorted, registered, big_depth = device.registration.apply(depth_frame,
                                                                   color_frame,
                                                                   with_big_depth=True)





    cv2.imshow("1", np.array(registered.to_image()))
    # points = device.registration.get_points_xyz_array(undistorted)
    # points = points[~np.isnan(points).any(axis=2)]
    # points = points.reshape(-1, 3) / 5

    # print(points.shape)


    # if np.isnan(points).any():
    #     print("nan")
    #     return
    #
    # reconstruction.points = o3d.utility.Vector3dVector(points)
    # vis.update_geometry(reconstruction)


    print("123")
    # global threads
    # threads.pop(thread_index)

running = True
device = Device()

depth_frame = None
color_frame = None

n_cpus = psutil.cpu_count()

with device.running():

    for type_, frame in device:

        if type_ == FrameType.Depth:
            depth_frame = frame
        elif type_ == FrameType.Color:
            color_frame = frame

        if depth_frame is not None and \
            color_frame is not None:
            #
            if len(threads) > 100:
                depth_frame = None
                color_frame = None
                continue

            undistorted, registered, big_depth = device.registration.apply(depth_frame,
                                                                           color_frame,
                                                                           with_big_depth=True)

            cv2.imshow('Grayscale', np.array(registered.to_image()))

            # cv2.imshow("1", np.array(registered.to_image()))
            # thread = Thread(target=parse_frame, args=(depth_frame, color_frame))
            # threads.append(thread)
            # thread.start()
            #
            # print(len(threads))
            #
            # depth_frame = None
            # color_frame = None

        if not vis.poll_events():
            break

        vis.update_renderer()

vis.close()

running = False

print("joining threads...")

for thread in threads:
    thread.join()
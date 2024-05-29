from freenect2 import Device, FrameType

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


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 5
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 10
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result




def update_points(color_frame, depth_frame):

    global reconstruction

    undistorted, registered, big_depth = device.registration.apply(frames[FrameType.Color], frames[FrameType.Depth],
                                                                   with_big_depth=True)

    points = device.registration.get_points_xyz_array(undistorted)

    points = points[~np.isnan(points).any(axis=2)]

    points = points.reshape(-1, 3) / 5
    #
    # num_points_to_select = int(0.1 * points.shape[0])
    # indices = np.random.choice(points.shape[0], num_points_to_select, replace=False)


    # points = points[indices]

    pcd = o3d.geometry.PointCloud()


    if np.isnan(points).any():
        print("nan")

    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd = pcd.uniform_down_sample(100)
    pcd = pcd.voxel_down_sample(voxel_size)

    # pcd.voxel_down_sample(voxel_size=0.05)

    # reconstruction.points = pcd.points
    # reconstruction.points = points

    if len(reconstruction.points) <= 5:
        reconstruction.points = pcd.points
        # visualise_sparse(reconstruction.points)
        return

    print(len(reconstruction.points))




    print(":: Sample mesh to point cloud")
    # draw_registration_result(prev, curr, np.identity(4))

    # source_down, source_fpfh = preprocess_point_cloud(reconstruction, voxel_size)
    # target_down, target_fpfh = preprocess_point_cloud(pcd, voxel_size)
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    #
    # pcd.transform(result_ransac.transformation)

    # reconstruction.points = pcd.points

    reconstruction.points.extend(pcd.points)

running = True

def compute():
    while running:
        if FrameType.Color in frames \
            and FrameType.Depth in frames \
            and FrameType.Color in frames:

            print("Computing")

            color_frame = frames[FrameType.Color]
            depth_frame = frames[FrameType.Depth]
            #
            # frames[FrameType.Color] = None
            # frames[FrameType.Depth] = None

            update_points(color_frame, depth_frame)

            vis.update_geometry(reconstruction)


device = Device()
frames = {}
threads = []

for i in range(10):
    threads.append(Thread(target=compute))
    print(i)
    threads[i].start()

with device.running():
    for type_, frame in device:

        # print("updated")
        frames[type_] = frame

        if not vis.poll_events():
            break
        # if not device.running():
        #     break

        vis.update_renderer()

vis.close()

running = False


for thread in threads:
    thread.join()


o3d.visualization.draw_geometries([reconstruction])
from threading import  Thread
import cv2
import numpy as np
import ktb

import open3d as o3d

voxel_size = 0.1
fitness_minimum = 0.9

vis = o3d.visualization.Visualizer()
vis.create_window()

reconstruction = o3d.geometry.PointCloud()
reconstruction.points = o3d.utility.Vector3dVector(np.random.rand(2, 3))


# vis.add_geometry(pcd)
vis.add_geometry(reconstruction)

def get_pcd(k):
    points, colors = k.get_ptcld(colorized=True, scale=1000)

    points = points.reshape((-1, 3))
    colors = colors.reshape((-1, 3))

    colors[:, [0, 2]] = colors[:, [2, 0]]

    points = o3d.utility.Vector3dVector(points)
    colors = o3d.utility.Vector3dVector(colors)

    pcd = o3d.geometry.PointCloud()

    pcd.points = points
    pcd.colors = colors

    pcd.points = pcd.voxel_down_sample(voxel_size=voxel_size).points

    return pcd

def calc_transformation(cur_pcd, prev_pcd, cur_fpfh, prev_fpfh):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        cur_pcd, prev_pcd, cur_fpfh, prev_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4, checkers=[],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999)
    )
    print(result.fitness)

running = True

def run_loop():
    k = ktb.Kinect()

    prev_pcd = None
    prev_fpfh = None
    prev_normals = None

    while running:
        cur_pcd = get_pcd(k)

        cur_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        cur_fpfh = o3d.pipelines.registration.compute_fpfh_feature(cur_pcd,
                                                      o3d.geometry.KDTreeSearchParamHybrid(radius=0.25,
                                                                                                       max_nn=100))
        if prev_pcd is None:
            reconstruction.points = cur_pcd.points
            reconstruction.paint_uniform_color([0.8, 0.8, 0])
            vis.update_geometry(reconstruction)
        else:
            # thread = Thread(target=calc_transformation, args=(cur_pcd, prev_pcd, cur_fpfh, prev_fpfh))
            # thread.start()

            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                cur_pcd, prev_pcd, cur_fpfh, prev_fpfh,
                mutual_filter=True,
                max_correspondence_distance=voxel_size,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4, checkers=[],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999)
            )


            # print(result.transformation)
            # eg_p2p = o3d.pipelines.registration.registration_icp(
            #     reconstruction, pcd, threshold, trans_init,
            #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
            # if result.fitness > fitness_minimum:
            reconstruction.points.extend(cur_pcd.transform(-result.transformation).points)
            vis.update_geometry(reconstruction)

        prev_pcd = cur_pcd
        prev_fpfh = cur_fpfh

        print("update")


t = Thread(target=run_loop)
t.start()

while running:
    # vis.update_geometry(reconstruction)
    # print("E")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    running = vis.poll_events()
    vis.update_renderer()

t.join()
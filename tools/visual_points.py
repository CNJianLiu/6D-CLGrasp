import open3d as o3d
import numpy as np
from functools import singledispatch

# @singledispatch
# def visual_points(points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.paint_uniform_color([0, 0, 1.0])
#     FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
#     o3d.visualization.draw_geometries([FOR1, pcd])

# @visual_points.register
# def _(points1, points2):
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(points1)
#     pcd1.paint_uniform_color([0, 0, 1.0])
#     pcd2 = o3d.geometry.PointCloud()
#     pcd2.points = o3d.utility.Vector3dVector(points2)
#     pcd2.paint_uniform_color([0, 1.0, 0])
#     FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
#     o3d.visualization.draw_geometries([FOR1, pcd1, pcd2])

# @visual_points.register
def visual_points( points1=[], points2=[], points3=[], points4=[], points5=[]):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.paint_uniform_color([0, 0, 0])#red
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 1.0, 0])#green
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(points3)
    pcd3.paint_uniform_color([0, 0, 1.0])#bule
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(points4)
    pcd4.paint_uniform_color([0, 0, 0]) #bladk
    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(points5)
    pcd5.paint_uniform_color([1.0, 1.0, 0]) #yellow      
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4, pcd5, FOR1])



if __name__ == '__main__':
    points1 = np.random.rand(1000, 3)
    points2 = np.random.rand(500, 3)
    points3 = np.random.rand(500, 3)    
    visual_points()
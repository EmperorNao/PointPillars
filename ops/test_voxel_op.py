import torch
from voxel_op import hard_voxelize


def test_voxel_op():

    points = torch.FloatTensor(
        [[0.1, 0.0, 0.0], [0.3, 0.3, 0.0], [0.0, -0.1, 0.0], [2.1, 2.1, 1.0], [2.0, 2.0, 1.0]]
    )

    voxel_size = [1.0, 1.0, 0.2]
    coors_range = [-2, -2, -2, 2, 2, 2]
    max_points=2
    max_voxels=10

    voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
    coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
    num_points_per_voxel = points.new_zeros(
        size=(max_voxels, ), dtype=torch.int)
    voxel_num = hard_voxelize(points, voxels, coors,
                                num_points_per_voxel, voxel_size,
                                coors_range, max_points, max_voxels, 3)
    print(voxel_num)


if __name__ == "__main__":
    test_voxel_op()
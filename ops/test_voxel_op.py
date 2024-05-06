import torch
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from voxel_op import hard_voxelize


def test_voxel_op():
    
    points = torch.FloatTensor(np.fromfile("ops/0000000000.bin", dtype=np.float32).reshape(-1, 4))[:, :3]

    voxel_size = [1.0, 1.0, 0.5]
    coors_range = [-50, -50, -4, 50, 50, 4]
    max_points=10
    max_voxels=10000

    voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
    coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
    num_points_per_voxel = points.new_zeros(
        size=(max_voxels, ), dtype=torch.int)
    voxel_num = hard_voxelize(points, voxels, coors,
                                num_points_per_voxel, voxel_size,
                                coors_range, max_points, max_voxels, 3)
    
    img = np.zeros((100, 100))

    for c in coors:
        img[c[1]][c[2]] = 1

    plt.figure(figsize=(16, 10))
    plt.scatter(points[:, 0] + 50, points[:, 1] + 50, s=1.0)    
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
    test_voxel_op()
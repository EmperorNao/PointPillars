import torch


def hard_voxelize(
        points,
        voxels, # returned
        coords, # returned 
        num_points_per_voxel, # returned
        voxel_size,
        coors_range,
        max_points,
        max_voxels,
        NDim = 3, determ=False):
    """
        return num voxels
    """
    return hard_voxelize_cpu(points,
        voxels,
        coords,
        num_points_per_voxel,
        voxel_size,
        coors_range,
        max_points,
        max_voxels,
        NDim)


def hard_voxelize_cpu(        points,
        voxels, # returned
        coors, # returned 
        num_points_per_voxel, # returned
        voxel_size,
        coors_range,
        max_points,
        max_voxels,
        NDim = 3):
  
    num_points = points.shape[0]
    num_features = points.shape[1]
    grid_size = [0] * NDim
    for i in range(0, NDim):
        grid_size[i] = round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i])

  
    coor_to_voxelidx = -torch.ones([grid_size[2], grid_size[1], grid_size[0]]).to(device=points.device)

    voxel_num = hard_voxelize_kernel(
        points, 
        voxels,
        coors, 
        num_points_per_voxel,
        coor_to_voxelidx, 
        voxel_size,
        coors_range, 
        grid_size, 
        max_points, 
        max_voxels, 
        num_points,
        num_features, 
        NDim
    )

    return voxel_num


def hard_voxelize_kernel(
    points,
    voxels,
    coors,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxel_size,
    coors_range,
    grid_size,
    max_points, 
    max_voxels,
    num_points, 
    num_features,
    NDim
    ):
  
    voxel_num = 0

    temp_coors = torch.zeros(
      [num_points, NDim]).to(device=points.device)

    dynamic_voxelize_kernel(points, 
                            temp_coors,
                            voxel_size, 
                            coors_range, 
                            grid_size,
                            num_points, 
                            num_features, 
                            NDim)

    coor = temp_coors

    for i in range(0, num_points):

        if coor[i][0] == -1: 
            continue

        def as_idx(el):
            return int(el)

        voxelidx = coor_to_voxelidx[as_idx(coor[i][0]), as_idx(coor[i][1]), as_idx(coor[i][2])]

        voxelidx = as_idx(voxelidx)
        if (as_idx(voxelidx) == -1):
            voxelidx = voxel_num
            if (max_voxels != -1 and voxel_num >= max_voxels):
                continue
            voxel_num += 1

            coor_to_voxelidx[as_idx(coor[i][0]), as_idx(coor[i][1]), as_idx(coor[i][2])] = voxelidx

            for k in range(0, NDim):
                coors[voxelidx][k] = coor[i][k]

        num = num_points_per_voxel[voxelidx]
        if (max_points == -1 or num < max_points):
            for k in range(0, num_features):
                voxels[voxelidx][num][k] = points[i][k]
        num_points_per_voxel[voxelidx] += 1

    return voxel_num


def dynamic_voxelize_kernel(
        points,
        coors,
        voxel_size,
        coors_range,
        grid_size,
        num_points, 
        num_features,
        NDim):
    ndim_minus_1 = NDim - 1
    from math import floor
    failed = False
    coor = [0] * 3

    for i in range(0, num_points):
        failed = False
        for j in range(0, NDim):
            c = floor((points[i][j] - coors_range[j]) / voxel_size[j])
            if ((c < 0 or c >= grid_size[j])):
                failed = True
                break
            coor[ndim_minus_1 - j] = c

        for k in range(0, NDim):
            if (failed):
                coors[i][k] = -1
            else:
                coors[i][k] = coor[k]
    
    return

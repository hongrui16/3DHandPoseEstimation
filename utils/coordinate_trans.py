import torch
import numpy as np



def camera_xyz_to_uv(xyz, intrinsic_matrix):
    """
    Convert 3D coordinates to 2D coordinates using a camera's intrinsic matrix.

    Args:
        xyz: A tensor of 3D coordinates, shape (N, 3), where N is the number of points.
        intrinsic_matrix: A tensor representing the camera's intrinsic matrix, shape (3, 3).

    Returns:
        A tensor of 2D coordinates, shape (N, 2).
    """

    # print(f'xyz.shape: {xyz.shape}')
    # print(f'intrinsic_matrix.shape: {intrinsic_matrix.shape}')
    uvw = torch.matmul(xyz, intrinsic_matrix.t())
    # print('uvw\n', uvw)

    # Normalize by the last coordinate to get (u, v)
    uv = uvw[:, :2] / uvw[:, 2].unsqueeze(1)

    return uv

def batch_xyz_to_uv(xyz_batch, K):
    

    # Convert the shape of points_3d from (bs, num, 3) to (bs*num, 3)
    bs, num, _ = xyz_batch.shape
    points_3d_reshaped = xyz_batch.view(bs * num, 3)

    # Use matrix multiplication to multiply the camera intrinsic parameter matrix K and the three-dimensional coordinate points_3d
    p = torch.matmul(K, points_3d_reshaped.t()) # Note that transposition is required

    #Normalize to get the two-dimensional coordinates (u, v, 1). Note that you need to divide by the last row.
    uv = p[:-1] / p[-1]

    # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
    uv = uv.t().view(bs, num, 2)

    return uv


def batch_project_xyz_to_uv(positions_xyz, camera_intrinsic_matrix):
    """
    Projects three-dimensional coordinates onto the image plane.

    Args:
        positions_xyz: Three-dimensional coordinates, shape (batch_size, num_points, 3).
        camera_intrinsic_matrix: Camera intrinsic parameter matrix, shape is (batch_size, 3, 3).
    Returns:
        The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
    """
    # print(f'positions_xyz.shape: {positions_xyz.shape}') #torch.Size([2, 21, 3]
    # print(f'camera_intrinsic_matrix.shape: {camera_intrinsic_matrix.shape}') #torch.Size([2, 21, 3]
    # print('positions_xyz[0]', positions_xyz[0])
    

    # bs, num_points, _ = positions_xyz.shape

    # Adjust the shape of positions_xyz to match camera_intrinsic_matrix
    # Reshape to (bs, 3, num_points)
    points_3d_reshaped = positions_xyz.permute(0, 2, 1)
    # points_3d_reshaped shape is [bs, 3, num_points]


    # Use batch matrix multiplication
    # camera_intrinsic_matrix shape is [bs, 3, 3]
    p = torch.bmm(camera_intrinsic_matrix, points_3d_reshaped)

    # The shape of p should now be [bs, 3, num_points]

    # Check if the last row of p has any zero values and replace with a small non-zero value to avoid dividing by zero
    p[:, -1, :] = torch.where(p[:, -1, :] == 0, torch.tensor(1e-10, dtype=p.dtype, device=p.device), p[ :, -1, :])
    
    # print(f'p.shape: {p.shape}') # should be [bs, num_points, 2]

    #Normalize to get final 2D coordinates
    #The shape becomes [bs, num_points, 2]
    uv = (p[:, :2, :] / p[:, -1, :].unsqueeze(1)).permute(0, 2, 1)

    # print(f'uv.shape: {uv.shape}') # should be [bs, num_points, 2]

    # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
    # uv = uv.t().view(bs, num_points, 2)
    # print('uv[0]', uv[0])

    return uv

if __name__ == '__main__':

        
    # Example usage
    intrinsic_matrix = torch.tensor([[320, 0.0000, 160.0000],
                                    [0.0000, 320, 160.0000],
                                    [0.0000, 0.0000, 1.0000]])

    # xyz = torch.tensor([[0.0, 0.0, 0.0],
    #                     [1.0, 2.0, 3.0],
    #                     [4.0, 5.0, 6.0]])

    xyz = torch.tensor([
            [ 0.5, -0.5,  1],
            [ 1,  1,  2],
            [ 0.0206,  0.0136,  0.6067]
        ])
        
    
    # # Define three-dimensional coordinates points_3d, the shape is (bs, num, 3)
    # points_3d_batch = torch.tensor([
    #     [[0, 0, 0],
    #     [0.5, 1.0, 2.0],
    #     [0.5, 0.5, 2.0]],

    #     [[-1.0, 0.0, 3.0],
    #     [1.0, 0.0, 3.0],
    #     [1.0, 1.0, 3.0]]
    # ], dtype=torch.float32)
    
    # 

    # uv = batch_xyz_to_uv(points_3d_batch, intrinsic_matrix)
    uv = camera_xyz_to_uv(xyz, intrinsic_matrix)
    print(uv)


    b_uv = batch_project_xyz_to_uv(xyz.unsqueeze(0), intrinsic_matrix.unsqueeze(0))
    print(b_uv)

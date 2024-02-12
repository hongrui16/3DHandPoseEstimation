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




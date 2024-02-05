import torch
import numpy as np



def xyz_to_uv(xyz, intrinsic_matrix):
    """
    Convert 3D coordinates to 2D coordinates using a camera's intrinsic matrix.

    Args:
        xyz: A tensor of 3D coordinates, shape (N, 3), where N is the number of points.
        intrinsic_matrix: A tensor representing the camera's intrinsic matrix, shape (3, 3).

    Returns:
        A tensor of 2D coordinates, shape (N, 2).
    """

    # Convert xyz to homogeneous coordinates (add a dimension of 1s)
    ones = torch.ones(xyz.shape[0], 1)
    xyz_homogeneous = torch.cat((xyz, ones), dim=1)
    # print('xyz\n', xyz)
    # print('xyz_homogeneous\n', xyz_homogeneous)
    # print('intrinsic_matrix\n', intrinsic_matrix)
    # print('intrinsic_matrix.t()\n', intrinsic_matrix.t())
    # Apply the intrinsic matrix (3x4 matrix multiplication)
    uvw = torch.dot(xyz_homogeneous, intrinsic_matrix.t())
    print('uvw\n', uvw)

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

def convert_camera_to_pixel_coordinates(points_3d, intrinsic_matrix):
    """
    Convert 3D points in camera coordinates to 2D pixel coordinates using OpenCV.

    Args:
        points_3d (numpy.ndarray): 3D points in camera coordinates, shape (N, 3).
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix, shape (3, 3).

    Returns:
        numpy.ndarray: 2D points in pixel coordinates, shape (N, 2).
    """
    # 将 3D 点扩展为齐次坐标
    # ones = torch.ones((points_3d.shape[0], 1))
    # points_3d_homogeneous = torch.hstack((points_3d, ones))
    
    # print('points_3d_homogeneous.shape', points_3d_homogeneous.shape)
    # print('points_3d_homogeneous\n', points_3d_homogeneous)
    # 使用内参矩阵投影到 2D
    points_2d_homogeneous = torch.matmul(points_3d, intrinsic_matrix.T)

    # 从齐次坐标中提取出 (u, v)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    return points_2d


if __name__ == '__main__':

        
    # Example usage
    intrinsic_matrix = torch.tensor([[282.9000, 0.0000, 160.0000],
                                    [0.0000, 282.9000, 160.0000],
                                    [0.0000, 0.0000, 1.0000]])

    # xyz = torch.tensor([[0.0, 0.0, 0.0],
    #                     [1.0, 2.0, 3.0],
    #                     [4.0, 5.0, 6.0]])

    xyz = torch.tensor([
            [ 0.0049, -0.0572,  0.7018],
            [ 0.0141,  0.0303,  0.5829],
            [ 0.0206,  0.0136,  0.6067]
        ])
        
    
    uv = convert_camera_to_pixel_coordinates(xyz, intrinsic_matrix)
    print(uv)
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

    # uv = xyz_to_uv(points_3d_batch, intrinsic_matrix)
    # print(uv)




import torch
import numpy as np

def xyz2uv_torch():
     # Define camera internal parameter matrix K (shape: 3x3)
     K = torch.tensor([[640, 0, 320],
                     [0, 640, 320],
                     [0, 0, 1]], dtype=torch.float32)

     # Define the coordinates of a three-dimensional point with a shape of 3x1
     point_3d = torch.tensor([[0.5615],
                             [-0.2014],
                             [0.7949]], dtype=torch.float32)

     # Define the coordinates of a three-dimensional point with a shape of 3x1
     point_3d = torch.tensor([[1.615],
                             [-2.2014],
                             [2.7949]], dtype=torch.float32)

     # Multiply the camera internal parameter matrix K and the matrix of the three-dimensional point point_3d
     p = torch.matmul(K, point_3d)

     # Normalize to get (u, v, 1), note that you need to divide by the last row
     uv = p[:-1] / p[-1]

     print(uv)
    # tensor([[772.0820],
    # [157.8463]])

def batch_xyz2uv_torch():
     K = torch.tensor([[640, 0, 320],
                     [0, 640, 320],
                     [0, 0, 1]], dtype=torch.float32)

     # Define three-dimensional coordinates points_3d, the shape is (bs, num, 3)
     points_3d = torch.tensor([
         [[-0.5, 1.0, 2.0],
          [0.5, 1.0, 2.0],
          [0.5, 0.5, 2.0]],
        
         [[-1.0, 0.0, 3.0],
          [1.0, 0.0, 3.0],
          [1.0, 1.0, 3.0]]
     ], dtype=torch.float32)

     # Convert the shape of points_3d from (bs, num, 3) to (bs*num, 3)
     bs, num, _ = points_3d.shape
     points_3d_reshaped = points_3d.view(bs * num, 3)

     # Use matrix multiplication to multiply the camera intrinsic parameter matrix K and the three-dimensional coordinate points_3d
     p = torch.matmul(K, points_3d_reshaped.t()) # Note that transposition is required

     #Normalize to get the two-dimensional coordinates (u, v, 1). Note that you need to divide by the last row.
     uv = p[:-1] / p[-1]

     # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
     uv = uv.t().view(bs, num, 2)

     print(uv)


def xyz2uv_numpy():

     # Camera matrix (example values, replace with your actual K matrix)
     K = np.array([[640, 0, 320],
                 [0, 640, 320],
                 [0, 0, 1]])

     # Camera coordinates (3D point in meters)
     Xc, Yc, Zc = 1.5, 0.8, 4.0 # Example values
    
     xyz = [Xc, Yc, Zc]

     xyz = np.array([[1.615],
      [-2.2014],
      [2.7949]]).squeeze().tolist()
     #Convert to pixel coordinates
     x_proj = K.dot(xyz)
     x, y = x_proj[:2] / x_proj[2]

     print("Pixel coordinates:", x, y)

if __name__ == '__main__':

     xyz2uv_numpy()
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np
import mano
from mano.utils import Mesh

import os, sys
# Get the absolute path of the parent of the parent directory
sys.path.append('../..')
from config import config

colors = {
    'pink': [1.00, 0.75, 0.80],
    'skin': [0.96, 0.75, 0.69],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)



def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todense())
    elif 'chumpy' in str(type(array)):
        array = np.array(array)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array.astype(dtype)




class ManoLayer(nn.Module):
    def __init__(self, device, MANO_RIGHT_pkl = None, bases_num = 10, pose_num = 6, mesh_num=778, keypoints_num=16):       
        super(ManoLayer, self).__init__()
         
        self.device = device
        self.bases_num = bases_num 
        self.pose_num = pose_num
        self.mesh_num = mesh_num
        self.keypoints_num = keypoints_num
        
        # self.dd = pickle.load(open('config/mano/models/MANO_RIGHT.pkl', 'rb'))
        self.dd = pickle.load(open(MANO_RIGHT_pkl, 'rb'), encoding='latin1')
        # print("self.dd['hands_components']", self.dd['hands_components'].shape) #(45, 45
        self.kintree_table = self.dd['kintree_table']
        self.id_to_col = {self.kintree_table[1,i] : i for i in range(self.kintree_table.shape[1])} 
        self.parent = {i : self.id_to_col[self.kintree_table[0,i]] for i in range(1, self.kintree_table.shape[1])}  

        self.mesh_mu = Variable(torch.from_numpy(np.expand_dims(self.dd['v_template'], 0).astype(np.float32)).to(device)) # zero mean
        self.mesh_pca = Variable(torch.from_numpy(np.expand_dims(self.dd['shapedirs'], 0).astype(np.float32)).to(device))
        self.posedirs = Variable(torch.from_numpy(np.expand_dims(self.dd['posedirs'], 0).astype(np.float32)).to(device))
        self.J_regressor = Variable(torch.from_numpy(np.expand_dims(self.dd['J_regressor'].todense(), 0).astype(np.float32)).to(device))
        self.weights = Variable(torch.from_numpy(np.expand_dims(self.dd['weights'], 0).astype(np.float32)).to(device))
        self.hands_components = Variable(torch.from_numpy(np.expand_dims(np.vstack(self.dd['hands_components'][:pose_num]), 0).astype(np.float32)).to(device))
        self.hands_mean       = Variable(torch.from_numpy(np.expand_dims(self.dd['hands_mean'], 0).astype(np.float32)).to(device))
        self.root_rot = Variable(torch.FloatTensor([np.pi,0.,0.]).unsqueeze(0).to(device))

        data_struct = Struct(**self.dd)

        self.faces = data_struct.f

    def rodrigues(self, r):       
        theta = torch.sqrt(torch.sum(torch.pow(r, 2),1))  

        def S(n_):   
            ns = torch.split(n_, 1, 1)     
            Sn_ = torch.cat([torch.zeros_like(ns[0]),-ns[2],ns[1],ns[2],torch.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],torch.zeros_like(ns[0])], 1)
            Sn_ = Sn_.view(-1, 3, 3)      
            return Sn_    

        n = r/(theta.view(-1, 1))   
        Sn = S(n) 

        #R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
        #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)
        
        I3 = Variable(torch.eye(3).unsqueeze(0).to(self.device))

        R = I3 + torch.sin(theta).view(-1, 1, 1)*Sn\
            +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

        Sr = S(r)
        theta2 = theta**2     
        R2 = I3 + (1.-theta2.view(-1,1,1)/6.)*Sr\
            + (.5-theta2.view(-1,1,1)/24.)*torch.matmul(Sr,Sr)
        
        idx = np.argwhere((theta<1e-30).data.cpu().numpy())

        if (idx.size):
            R[idx,:,:] = R2[idx,:,:]

        return R,Sn

    def get_poseweights(self, poses, bsize):
        # pose: batch x 24 x 3                                                    
        pose_matrix, _ = self.rodrigues(poses[:,1:,:].contiguous().view(-1,3))
        #pose_matrix, _ = rodrigues(poses.view(-1,3))    
        pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(self.keypoints_num-1),axis=0)).to(self.device))
        pose_matrix = pose_matrix.view(bsize, -1)
        return pose_matrix

    def rot_pose_beta_to_mesh(self, rots, poses, betas):

        batch_size = rots.size(0)   

        poses = (self.hands_mean + torch.matmul(poses.unsqueeze(1), self.hands_components).squeeze(1)).view(batch_size,self.keypoints_num-1,3)
        #poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)   
        poses = torch.cat((self.root_rot.repeat(batch_size,1).view(batch_size,1,3),poses),1)

        v_shaped =  (torch.matmul(betas.unsqueeze(1), 
                    self.mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,self.bases_num,-1)).squeeze(1)    
                    + self.mesh_mu.repeat(batch_size,1,1).view(batch_size, -1)).view(batch_size, self.mesh_num, 3)      
        
        pose_weights = self.get_poseweights(poses, batch_size)    

        v_posed = v_shaped + torch.matmul(self.posedirs.repeat(batch_size,1,1,1),
                (pose_weights.view(batch_size,1,(self.keypoints_num - 1)*9,1)).repeat(1,self.mesh_num,1,1)).squeeze(3)

        J_posed = torch.matmul(v_shaped.permute(0,2,1),self.J_regressor.repeat(batch_size,1,1).permute(0,2,1))
        J_posed = J_posed.permute(0, 2, 1)
        J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]
            
        pose = poses.permute(1, 0, 2)
        pose_split = torch.split(pose, 1, 0)


        angle_matrix =[]
        for i in range(self.keypoints_num):
            out, tmp = self.rodrigues(pose_split[i].contiguous().view(-1, 3))
            angle_matrix.append(out)

        #with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

        with_zeros = lambda x:\
            torch.cat((x,   Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1).to(self.device))  ),1)

        pack = lambda x: torch.cat((Variable(torch.zeros(batch_size,4,3).to(self.device)),x),2) 

        results = {}
        results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size,3,1)),2))

        for i in range(1, self.kintree_table.shape[1]):
            tmp = with_zeros(torch.cat((angle_matrix[i],
                            (J_posed_split[i] - J_posed_split[self.parent[i]]).view(batch_size,3,1)),2)) 
            results[i] = torch.matmul(results[self.parent[i]], tmp)

        results_global = results

        results2 = []
            
        for i in range(len(results)):
            vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size,1).to(self.device)) ),1)).view(batch_size,4,1)
            results2.append((results[i]-pack(torch.matmul(results[i], vec))).unsqueeze(0))    

        results = torch.cat(results2, 0)
        
        T = torch.matmul(results.permute(1,2,3,0), self.weights.repeat(batch_size,1,1).permute(0,2,1).unsqueeze(1).repeat(1,4,1,1))
        Ts = torch.split(T, 1, 2)
        rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size,self.mesh_num,1).to(self.device)) ), 2)  
        rest_shape_hs = torch.split(rest_shape_h, 1, 2)

        v = Ts[0].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, self.mesh_num)\
            + Ts[1].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, self.mesh_num)\
            + Ts[2].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, self.mesh_num)\
            + Ts[3].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, self.mesh_num)
    
        #v = v.permute(0,2,1)[:,:,:3] 
        Rots = self.rodrigues(rots)[0]

        Jtr = []

        for j_id in range(len(results_global)):
            Jtr.append(results_global[j_id][:,:3,3:4])

        # Add finger tips from mesh to joint list    
        Jtr.insert(4,v[:,:3,333].unsqueeze(2))
        Jtr.insert(8,v[:,:3,444].unsqueeze(2))
        Jtr.insert(12,v[:,:3,672].unsqueeze(2))
        Jtr.insert(16,v[:,:3,555].unsqueeze(2))
        Jtr.insert(20,v[:,:3,745].unsqueeze(2))        
        
        Jtr = torch.cat(Jtr, 2) #.permute(0,2,1)
            
        vertices = torch.matmul(Rots,v[:,:3,:]).permute(0,2,1) #.contiguous().view(batch_size,-1)
        joint = torch.matmul(Rots,Jtr).permute(0,2,1) #.contiguous().view(batch_size,-1)
        # print('v shape:', v.shape)
        # print('Jtr shape:', Jtr.shape)
        return vertices, joint


    
    def hand_meshes(self, vertices, vc= colors['skin']):

        if vertices.ndim <3:
            vertices = vertices.reshape(-1,778,3)
        vertices = to_np(vertices)

        meshes = []
        for v in vertices:
            hand_mesh = Mesh(vertices=v, faces=self.faces, vc=vc)
            meshes.append(hand_mesh)

        return  meshes

    def joint_meshes(self, joints, radius=.002, vc=colors['green']):

        joints = to_np(joints)
        if joints.ndim <3:
            joints = joints.reshape(1,-1,3)

        meshes = []
        for j in joints:
            joint_mesh = Mesh(vertices=j, radius=radius, vc=vc)
            meshes.append(joint_mesh)

        return  meshes

    def forward(self, root_angles, other_angles, betas):
        vertices, joint = self.rot_pose_beta_to_mesh(root_angles, other_angles, betas)
        return vertices, joint



def build_sequtial(input_dim, devide, output_dim):
    sequential = [] # Create an empty list to store layers

    # Calculate the minimum number of times that dimensionality can be reduced until the dimensionality reduction result is no less than output_dim
    quotient = 0
    devide = 4
    temp_dim = input_dim
    while temp_dim // devide >= output_dim:
        temp_dim //= devide
        quotient += 1

    # Gradually reduce the dimensionality, each time reducing it to half of the original value
    for i in range(quotient):
        next_dim = input_dim // (devide**(i+1)) # Calculate the dimensions of the next layer
        sequential.append(torch.nn.Linear(input_dim // (devide**i), next_dim))
        sequential.append(torch.nn.ReLU())

    # Ensure that the output dimension of the last layer is not less than output_dim
    # If quotient is 0, it means that input_dim itself is less than or equal to output_dim and should be directly connected to output_dim
    if quotient > 0:
        last_dim = input_dim // (devide**quotient)
    else:
        last_dim = input_dim
    # Add the last layer, the output dimension is output_dim
    sequential.append(torch.nn.Linear(last_dim, output_dim))
    sequential.append(torch.nn.Sigmoid())
    return sequential


class MANOBetasPrediction(torch.nn.Module):
    def __init__(self, device = 'cpu', input_dim = None):
        super(MANOBetasPrediction, self).__init__()        
        sequential = build_sequtial(input_dim, 4, config.mano_beta_num)
        #Create Sequential model
        self.mlp = torch.nn.Sequential(*sequential)

    def forward(self, x):
        betas = self.mlp(x)
        betas = betas - 0.5
        return betas


class MANOThetaPrediction(torch.nn.Module):
    def __init__(self, device = 'cpu', input_dim = None):
        super(MANOThetaPrediction, self).__init__()
        rot_dim = 3
        sequential = build_sequtial(input_dim, 4, rot_dim)
        #Create Sequential model
        self.mlp1 = torch.nn.Sequential(*sequential)

        sequential = build_sequtial(input_dim, 2, config.mano_pose_num)
        #Create Sequential model
        self.mlp2 = torch.nn.Sequential(*sequential)
        
    def forward(self, x):
        root_angles = self.mlp1(x)
        # Scale root_angles to the range of [-π, π]
        root_angles = (root_angles - 0.5)* 2 * math.pi

        other_angles = self.mlp2(x)
        # Scale other_angles to the range of [-π/2, π/2]
        other_angles = (other_angles - 0.5) * math.pi


        return root_angles, other_angles
    




if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MANO_RIGHT_pkl = '../../config/mano/models/MANO_RIGHT.pkl'
    n_comps = 45
    batch_size = 1
    model = ManoLayer(device, MANO_RIGHT_pkl, pose_num=n_comps)

    betas = torch.rand(batch_size, 10, device = device)  - 0.5
    pose = (torch.rand(batch_size, n_comps, device = device) - 0.5)*np.pi
    global_orient = (torch.rand(batch_size, 3, device = device) - 0.5) *2* np.pi
    
    # print('betas:', betas)
    # print('pose:', pose)


    vertices, joint = model.rot_pose_beta_to_mesh(global_orient, pose, betas)
    # print('vertices:', vertices)
    print('joint:', joint.shape)
    trans = torch.tensor([128, 128], device=device).view(1, 1, 2)  # add necessary dimensions
    scale = torch.tensor([540], device=device).view(1, 1, 1)  # add necessary dimensions

    joint_2d = trans + scale * joint[:,:,:2]



    # print('joint_2d', joint_2d)
    vertices_mesh = model.hand_meshes(vertices)
    joint_mesh = model.joint_meshes(joint)

    vertices_mesh[0].show()

    #visualize joints mesh only
    joint_mesh[0].show()

    #visualize hand and joint meshes
    # hj_meshes = Mesh.concatenate_meshes([vertices_mesh[0], joint_mesh[0]])
    # hj_meshes = Mesh.concatenate_meshes([joint_mesh[0]])
    # hj_meshes = Mesh.concatenate_meshes([vertices_mesh[0]])
    # hj_meshes.show() 

    # print('Finished')

    input_dim = 512
    input_feat = (torch.rand(batch_size, input_dim, device = device) - 0.5) *6
    betas_model = MANOBetasPrediction(device, input_dim).to(device)
    theta_model = MANOThetaPrediction(device, input_dim).to(device)
    
    betas = betas_model(input_feat)
    print('betas:', betas)
    rot, other = theta_model(input_feat)
    vertices, joint = model.rot_pose_beta_to_mesh(rot, other, betas)

    vertices_mesh = model.hand_meshes(vertices)
    joint_mesh = model.joint_meshes(joint)

    vertices_mesh[0].show()

    #visualize joints mesh only
    joint_mesh[0].show()

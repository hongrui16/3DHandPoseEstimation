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




'''


#-------------------
# Resnet + Mano
#-------------------

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out
    
class ResNet_Mano(nn.Module):

    def __init__(self, block, layers, input_option, num_classes=1000):

        self.input_option = input_option
        self.inplanes = 64
        super(ResNet_Mano, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)       
        #if (self.input_option):        
        self.conv11 = nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)                        
        self.mean = Variable(torch.FloatTensor([545.,128.,128.,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]).cuda())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
       
        if (self.input_option):       
            x = self.conv11(x)
        else:
            x = self.conv1(x[:,0:3])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 

        xs = self.fc(x)
        xs = xs + self.mean  

        scale = xs[:,0]
        trans = xs[:,1:3] 
        rot = xs[:,3:6]    
        theta = xs[:,6:12]
        beta = xs[:,12:] 

        x3d = rot_pose_beta_to_mesh(rot,theta,beta)
        
        x = trans.unsqueeze(1) + scale.unsqueeze(1).unsqueeze(2) * x3d[:,:,:2] 
        x = x.view(x.size(0),-1)      
              
        #x3d = scale.unsqueeze(1).unsqueeze(2) * x3d
        #x3d[:,:,:2]  = trans.unsqueeze(1) + x3d[:,:,:2] 
        
        return x, x3d

def resnet34_Mano(pretrained=False,input_option=1, **kwargs):
    
    model = ResNet_Mano(BasicBlock, [3, 4, 6, 3], input_option, **kwargs)    
    model.fc = nn.Linear(512 * 1, 22)

    return model

'''

if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MANO_RIGHT_pkl = '../../config/mano/models/MANO_RIGHT.pkl'
    n_comps = 45
    batch_size = 1
    model = ManoLayer(device, MANO_RIGHT_pkl, pose_num=n_comps)

    betas = torch.rand(batch_size, 10, device = device)*.1
    pose = torch.rand(batch_size, n_comps, device = device)*.1
    global_orient = torch.rand(batch_size, 3, device = device)
    
    


    vertices, joint = model.rot_pose_beta_to_mesh(global_orient, pose, betas)

    trans = torch.tensor([128, 128], device=device).view(1, 1, 2)  # add necessary dimensions
    scale = torch.tensor([540], device=device).view(1, 1, 1)  # add necessary dimensions

    joint_2d = trans + scale * joint[:,:,:2]



    print('joint_2d', joint_2d)
    vertices_mesh = model.hand_meshes(vertices)
    joint_mesh = model.joint_meshes(joint)

    vertices_mesh[0].show()

    #visualize joints mesh only
    joint_mesh[0].show()

    #visualize hand and joint meshes
    hj_meshes = Mesh.concatenate_meshes([vertices_mesh[0], joint_mesh[0]])
    # hj_meshes = Mesh.concatenate_meshes([joint_mesh[0]])
    # hj_meshes = Mesh.concatenate_meshes([vertices_mesh[0]])
    hj_meshes.show() 

    # print('Finished')

    '''
    model_path = '../../config/mano/models/MANO_RIGHT.pkl'
    n_comps = 45
    batch_size = 10

    rh_model = mano.load(model_path=model_path,
                        is_rhand= True,
                        num_pca_comps=n_comps,
                        batch_size=batch_size,
                        flat_hand_mean=False)

    betas = torch.rand(batch_size, 10)*.1
    pose = torch.rand(batch_size, n_comps)*.1
    global_orient = torch.rand(batch_size, 3)
    transl        = torch.rand(batch_size, 3)

    output = rh_model(betas=betas,
                    global_orient=global_orient,
                    hand_pose=pose,
                    transl=transl,
                    return_verts=True,
                    return_tips = True)

    # print('output:', output)
    h_meshes = rh_model.hand_meshes(output)
    j_meshes = rh_model.joint_meshes(output)

    print('h_meshes:', h_meshes[0].vertices.shape)
    print('j_meshes:', j_meshes[0].vertices.shape)

    #visualize hand mesh only
    h_meshes[0].show()

    #visualize joints mesh only
    j_meshes[0].show()

    #visualize hand and joint meshes
    hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
    hj_meshes.show() 

    '''
    


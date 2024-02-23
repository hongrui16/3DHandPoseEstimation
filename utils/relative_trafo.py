import torch

def _stitch_mat_from_vecs(vector_list):
    """
    Stitches a given list of vectors into a 4x4 matrix in PyTorch.

    Input:
        vector_list: list of 16 tensors, which will be stitched into a matrix.
                     List contains matrix elements in a row-first fashion 
                     (m11, m12, m13, m14, m21, m22, m23, m24, ...).
                     The length of the vectors has to be the same, 
                     interpreted as batch dimension.
    """
    assert len(vector_list) == 16, "There have to be exactly 16 tensors in vector_list."

    # Ensure all vectors have the same batch size
    batch_size = vector_list[0].shape[0]
    for vec in vector_list:
        assert vec.shape[0] == batch_size, "All vectors must have the same batch size."

    # Reshape and stack the vectors
    vector_list = [x.view(1, batch_size) for x in vector_list]
    trafo_matrix = torch.cat(vector_list, 0).view(4, 4, batch_size)

    # Transpose to get the desired shape
    trafo_matrix = trafo_matrix.permute(2, 0, 1)

    return trafo_matrix

def _atan2(y, x):
    """
    Implementation of atan2 in PyTorch. Returns in -pi .. pi.
    """
    # PyTorch's atan2 already handles the quadrant checks and returns values in the range -pi to pi
    return torch.atan2(y, x + 1e-8)

def _get_rot_mat_x_hom(angle):
    one_vec = torch.ones_like(angle)
    zero_vec = torch.zeros_like(angle)
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec, zero_vec,
                                                 zero_vec, torch.cos(angle), -torch.sin(angle), zero_vec,
                                                 zero_vec, torch.sin(angle), torch.cos(angle), zero_vec,
                                                 zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix

def _get_rot_mat_y_hom(angle):
    one_vec = torch.ones_like(angle)
    zero_vec = torch.zeros_like(angle)
    trafo_matrix = _stitch_mat_from_vecs([torch.cos(angle), zero_vec, torch.sin(angle), zero_vec,
                                                 zero_vec, one_vec, zero_vec, zero_vec,
                                                 -torch.sin(angle), zero_vec, torch.cos(angle), zero_vec,
                                                 zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix

def _get_rot_mat_z_hom(angle):
    one_vec = torch.ones_like(angle)
    zero_vec = torch.zeros_like(angle)
    trafo_matrix = _stitch_mat_from_vecs([torch.cos(angle), -torch.sin(angle), zero_vec, zero_vec,
                                                 torch.sin(angle), torch.cos(angle), zero_vec, zero_vec,
                                                 zero_vec, zero_vec, one_vec, zero_vec,
                                                 zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix


def _get_trans_mat_hom(trans):
    one_vec = torch.ones_like(trans)
    zero_vec = torch.zeros_like(trans)
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec, zero_vec,
                                                 zero_vec, one_vec, zero_vec, zero_vec,
                                                 zero_vec, zero_vec, one_vec, trans,
                                                 zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix


def _to_hom(vector):
    s = vector.size()
    vector = vector.view(s[0], -1, 1)
    ones = torch.ones((s[0], 1, 1), dtype=vector.dtype, device=vector.device)
    vector = torch.cat([vector, ones], 1)
    return vector

def _from_hom(vector):
    s = vector.size()
    vector = vector.view(s[0], -1, 1)
    return vector[:, :-1, :]



def _forward(length, angle_x, angle_y, T):
    # Update current transformation from local -> new local
    T_this = torch.matmul(_get_trans_mat_hom(-length), 
                          torch.matmul(_get_rot_mat_x_hom(-angle_x), _get_rot_mat_y_hom(-angle_y)))

    # Transformation from global -> new local
    T = torch.matmul(T_this, T)

    # Calculate global location of this point
    x0 = _to_hom(torch.zeros((length.shape[0], 3, 1)))
    x = torch.matmul(torch.inverse(T), x0)
    return x, T


def _backward(delta_vec, T):
    # Calculate length directly
    length = torch.sqrt(delta_vec[:, 0, 0]**2 + delta_vec[:, 1, 0]**2 + delta_vec[:, 2, 0]**2)

    # Calculate y rotation
    angle_y = _atan2(delta_vec[:, 0, 0], delta_vec[:, 2, 0])

    # This vector is an intermediate result and always has x=0
    delta_vec_tmp = torch.matmul(_get_rot_mat_y_hom(-angle_y), delta_vec)

    # Calculate x rotation
    angle_x = _atan2(-delta_vec_tmp[:, 1, 0], delta_vec_tmp[:, 2, 0])

    # Update current transformation from local -> new local
    T_this = torch.matmul(_get_trans_mat_hom(-length), 
                          torch.matmul(_get_rot_mat_x_hom(-angle_x), _get_rot_mat_y_hom(-angle_y)))

    # Transformation from global -> new local
    T = torch.matmul(T_this, T)

    # Make them all batched scalars
    length = length.view(-1)
    angle_x = angle_x.view(-1)
    angle_y = angle_y.view(-1)
    return length, angle_x, angle_y, T


# Encodes how the kinematic chain goes; Is a mapping from child -> parent: dict[child] = parent
kinematic_chain_dict = {0: 'root',

                        4: 'root',
                        3: 4,
                        2: 3,
                        1: 2,

                        8: 'root',
                        7: 8,
                        6: 7,
                        5: 6,

                        12: 'root',
                        11: 12,
                        10: 11,
                        9: 10,

                        16: 'root',
                        15: 16,
                        14: 15,
                        13: 14,

                        20: 'root',
                        19: 20,
                        18: 19,
                        17: 18}

# order in which we will calculate stuff
kinematic_chain_list = [0,
                        4, 3, 2, 1,
                        8, 7, 6, 5,
                        12, 11, 10, 9,
                        16, 15, 14, 13,
                        20, 19, 18, 17]


def bone_rel_trafo(coords_xyz):
    """ Transforms the given real xyz coordinates into a bunch of relative frames.
        The frames are set up according to the kinematic chain. Each parent of the chain
        is the origin for the location of the next bone, where the z-axis is aligned with the bone
        and articulation is measured as rotations along the x- and y- axes.

        Inputs:
            coords_xyz: BxNx3 matrix, containing the coordinates for each of the N keypoints
    """
    # with tf.variable_scope('bone_rel_transformation'):
    coords_xyz = coords_xyz.view(-1, 21, 3)
    trafo_list = [None for _ in kinematic_chain_list]
    coords_rel_list = [0.0 for _ in kinematic_chain_list]

    # Iterate kinematic chain list (from root --> leaves)
    for bone_id in kinematic_chain_list:
        parent_id = kinematic_chain_dict[bone_id]

        if parent_id == 'root':
            # If there is no parent, global = local
            delta_vec = _to_hom(coords_xyz[:, bone_id, :].unsqueeze(1))
            T = _get_trans_mat_hom(torch.zeros_like(coords_xyz[:, 0, 0]))

            # Get articulation angles from bone vector
            results = _backward(delta_vec, T)

            # Save results
            coords_rel_list[bone_id] = torch.stack(results[:3], 1)
            trafo_list[bone_id] = results[3]

        else:
            T = trafo_list[parent_id]
            assert T is not None, 'Something went wrong.'

            # Calculate coords in local system
            x_local_parent = torch.matmul(T, _to_hom(coords_xyz[:, parent_id, :].unsqueeze(1)))
            x_local_child = torch.matmul(T, _to_hom(coords_xyz[:, bone_id, :].unsqueeze(1)))

            # Calculate bone vector in local coords
            delta_vec = x_local_child - x_local_parent
            delta_vec = _to_hom(delta_vec[:, :3, :].unsqueeze(1))

            # Get articulation angles from bone vector
            results = _backward(delta_vec, T)

            # Save results
            coords_rel_list[bone_id] = torch.stack(results[:3], 1)
            trafo_list[bone_id] = results[3]

    coords_rel = torch.stack(coords_rel_list, 1)

    return coords_rel


def bone_rel_trafo_inv(coords_rel):
    """ Assembles relative coords back to xyz coords. Inverse operation to bone_rel_trafo().

        Inputs:
            coords_rel: BxNx3 matrix, containing the coordinates for each of the N keypoints [length, angle_x, angle_y]
    """
    # with tf.variable_scope('assemble_bone_rel'):
    s = coords_rel.shape
    if len(s) == 2:
        coords_rel = coords_rel.unsqueeze(0)
        s = coords_rel.shape
    assert len(s) == 3, "Has to be a batch of coords."

    # List of results
    trafo_list = [None for _ in kinematic_chain_list]
    coords_xyz_list = [0.0 for _ in kinematic_chain_list]

    # Iterate kinematic chain list (from root --> leaves)
    for bone_id in kinematic_chain_list:
        parent_id = kinematic_chain_dict[bone_id]

        if parent_id == 'root':
            # If there is no parent, global = local
            T = _get_trans_mat_hom(torch.zeros_like(coords_rel[:, 0, 0]))

            # Get articulation angles from bone vector
            x, T = _forward(length=coords_rel[:, bone_id, 0],
                            angle_x=coords_rel[:, bone_id, 1],
                            angle_y=coords_rel[:, bone_id, 2],
                            T=T)

            # Save results
            coords_xyz_list[bone_id] = _from_hom(x).squeeze(2)
            trafo_list[bone_id] = T

        else:
            T = trafo_list[parent_id]
            assert T is not None, 'Something went wrong.'

            # Get articulation angles from bone vector
            x, T = _forward(length=coords_rel[:, bone_id, 0],
                            angle_x=coords_rel[:, bone_id, 1],
                            angle_y=coords_rel[:, bone_id, 2],
                            T=T)

            # Save results
            coords_xyz_list[bone_id] = _from_hom(x).squeeze(2)
            trafo_list[bone_id] = T

    coords_xyz = torch.stack(coords_xyz_list, 1)
    return coords_xyz
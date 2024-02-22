import torch

def build_sequtial(input_dim, output_dim, devide = 4):
    sequential = [] # Create an empty list to store layers

    # Calculate the minimum number of times that dimensionality can be reduced until the dimensionality reduction result is no less than output_dim
    quotient = 0
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
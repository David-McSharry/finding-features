import numpy as np
import torch


def vis_numerical_input(input):

    a = torch.argmax(input[:, 0:113], dim=1)
    b = torch.argmax(input[:, 113:226], dim=1)
    x = torch.argmax(input[:, 226:228], dim=1)
    y = torch.argmax(input[:, 228:230], dim=1)


    print("a: ", int(a))
    print("b: ", int(b))
    print("x: ", int(x))
    print("y: ", int(y))
    print('xor: ', int(torch.bitwise_xor(x, y)))
    print('output: ', int((a+b)%113 + torch.bitwise_xor(x, y)))



def get_xor_bit(input_tensor):
    """
    Helper function that takes a in raw input and returns a tensor of the xor bit
    """
    bit1_one_hot = input_tensor[:, 226:228]
    bit2_one_hot = input_tensor[:, 228:230]

    bit1 = torch.argmax(bit1_one_hot, dim=1)
    bit2 = torch.argmax(bit2_one_hot, dim=1)

    xor_bit = (bit1 + bit2) % 2

    return xor_bit


def hook_layer_activations(
        model: torch.nn.Module,
        input: torch.Tensor,
        layer_index: int
    ):
    activations = []

    def _get_activations(module, input, output):
        activations.append(output.detach())

    handle = model.layers[layer_index].register_forward_hook(_get_activations)
    
    _ = model(input)

    
    handle.remove()
    
    # # Remove all hooks from the model
    # for module in model.modules():
    #     module._forward_hooks.clear()
    #     module._forward_pre_hooks.clear()
    #     module._backward_hooks.clear()

    return activations[0]




def get_xor_steering_vector(X_batch, mlp, layer_index):

    xor_bits_for_batch = get_xor_bit(X_batch)
    X_batch_xor0 = X_batch[xor_bits_for_batch == 0]
    X_batch_xor1 = X_batch[xor_bits_for_batch == 1]

    smallest_dim = min(len(X_batch_xor0), len(X_batch_xor1))

    activtations_for_batch_xor0 = hook_layer_activations(mlp, X_batch_xor0, layer_index)[:smallest_dim]
    activations_for_batch_xor1 = hook_layer_activations(mlp, X_batch_xor1, layer_index)[:smallest_dim]

    activation_diff = activtations_for_batch_xor0 - activations_for_batch_xor1

    # average across samples in the batch
    steering_vector = torch.mean(activation_diff, dim=0)

    return steering_vector


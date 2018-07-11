import os
from random import getrandbits

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

import tensorflow as tf


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None, return_paths=False):
        self.folder = folder
        self.images = [i for i in os.listdir(folder) if is_image_file(i)]
        self.transform = transform
        self.return_paths = return_paths

    def __getitem__(self, item):
        path = self.images[item]
        abspath = os.path.join(self.folder, path)
        image = default_loader(abspath)
        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            return path, image
        return image

    def __len__(self):
        return len(self.images)


# https://gist.github.com/kingspp/3ec7d9958c13b94310c1a365759aa3f4
# Pyfunc Gradient Function
def _py_func_with_gradient(func, inp, Tout, stateful=True, name=None,
                           grad_func=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate random name in order to avoid conflicts with inbuilt names
    rnd_name = 'PyFuncGrad-' + '%0x' % getrandbits(30 * 4)

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def convert_pytorch_model_to_tf(model, device, out_dims=None):
    """
    Convert a pytorch model into a tensorflow op that allows backprop
    :param model: A pytorch nn.Model object
    :param out_dims: The number of output dimensions (classes) for the model
    :return: A model function that maps an input (tf.Tensor) to the
    output of the model (tf.Tensor)
    """
    torch_state = {
        'logits': None,
        'x': None,
    }
    if not out_dims:
        out_dims = list(model.modules())[-1].out_features

    def _fprop_fn(x_np):
        x_tensor = torch.tensor(x_np, requires_grad=True)
                
        torch_state['x'] = x_tensor
        torch_state['logits'] = model(x_tensor.to(device))
        return torch_state['logits'].cpu().detach().numpy()

    def _bprop_fn(x_np, grads_in_np):
        _fprop_fn(x_np)

        grads_in_tensor = torch.tensor(grads_in_np).to(device)

        # Run our backprop through our logits to our xs
        loss = torch.sum(torch_state['logits'] * grads_in_tensor)
        loss.backward()
        return torch_state['x'].grad.cpu().numpy()

    def _tf_gradient_fn(op, grads_in):
        return tf.py_func(_bprop_fn, [op.inputs[0], grads_in],
                          Tout=[tf.float32])

    def tf_model_fn(x_op):
        out = _py_func_with_gradient(_fprop_fn, [x_op], Tout=[tf.float32],
                                     stateful=True,
                                     grad_func=_tf_gradient_fn)[0]
        out.set_shape([None, out_dims])
        return out

    return tf_model_fn

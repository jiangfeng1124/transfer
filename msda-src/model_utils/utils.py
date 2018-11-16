import torch
from torch.autograd import Variable

# Ref implementation in Tensorflow:
# - github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/utils.py

def compute_pairwise_distances(x, y):
    """ Computes the squared pairwise Euclidean distances between x and y.

    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]

    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples]

    Raise:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    norm = lambda x: torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())

def gaussian_kernel_matrix(x, y, sigmas):
    """ Computes a Gaussian RBK between the samples of x and y.

    We create a sum of multiple gaussian kernels each having a width sigma_i.

    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    """
    beta = Variable(1. / (2. * (sigmas.unsqueeze(1))))

    dist = compute_pairwise_distances(x, y)

    s = torch.matmul(beta, dist.view(1, -1))

    return (torch.sum(torch.exp(-s), 0)).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    """ Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use kernel two sample estimate
    using the empirical mean of the two distributions.

    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
      GaussianKernelMatrix.

    Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
    """

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    cost = torch.clamp(cost, min=0)
    return cost


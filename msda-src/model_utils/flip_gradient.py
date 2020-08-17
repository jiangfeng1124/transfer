
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class GRL(autograd.Function):
    ''' Gradient Reverse Layer
    '''
    def __init__(self, gamma=1.0):
        super(GRL, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input * (-self.gamma)

    def set_gamma(self, gamma):
        self.gamma = gamma

flip_gradient = GRL(1)

if __name__ == '__main__':

    # flip_gradient = GRL(1)

    x = Variable(torch.rand([4, 5]))
    y = Variable(torch.LongTensor([0, 0, 1, 1]))
    layer1 = nn.Linear(5, 5)
    layer2 = nn.Linear(5, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(layer1.parameters(), lr=0.1)
    for i in range(10):
        hx = layer1(x)
        grl_hx = flip_gradient(hx)
        output = layer2(grl_hx)

        loss = criterion(output, y)
        print(loss)
        loss.backward()
        optimizer.step()


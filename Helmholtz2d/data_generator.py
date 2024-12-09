import torch
import numpy as np
from ground_truth import helmholtz_2d_exact_u, helmholtz_2d_source_term

''' Contents : 1. Generate Train data
               2. Generate Test data    '''

''' 1. Generate Train data '''
def generate_Helmholtz_2d_train_data(num_train, num_bc, a1, a2, coefficient):
    # colocation points
    yc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    xc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    with torch.no_grad():
        uc = helmholtz_2d_source_term(yc, xc, a1, a2, coefficient)
    # requires grad
    yc.requires_grad = True
    xc.requires_grad = True
    # boundary points
    north = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    west = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    south = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    east = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    
    north.requires_grad = True
    west.requires_grad = True
    south.requires_grad = True
    east.requires_grad = True

    yb = torch.cat([
        torch.ones((num_bc, 1), requires_grad = True), west,
        torch.ones((num_bc, 1), requires_grad = True) * -1, east
        ])
    xb = torch.cat([
        north, torch.ones((num_bc, 1), requires_grad = True) * -1,
        south, torch.ones((num_bc, 1), requires_grad = True)
        ])
    with torch.no_grad():
        ub = helmholtz_2d_exact_u(yb, xb, a1, a2)
    return yc, xc, uc, yb, xb, ub

def generate_Helmholtz_2d_test_data(num_test, a1, a2):
    # test points
    y = torch.linspace(-1, 1, num_test)
    x = torch.linspace(-1, 1, num_test)
    y, x = torch.meshgrid([y, x], indexing='ij')
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    y_test.requires_grad = True
    x_test.requires_grad = True
    with torch.no_grad():
        u_test = helmholtz_2d_exact_u(y_test, x_test, a1, a2)
    return y_test, x_test, u_test

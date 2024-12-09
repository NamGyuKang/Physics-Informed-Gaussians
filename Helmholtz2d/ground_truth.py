import numpy as np
import torch

def helmholtz_2d_exact_u(y, x, a1, a2):
    return torch.sin(a1*torch.pi*y) * torch.sin(a2*torch.pi*x)

def helmholtz_2d_source_term(y, x, a1, a2, coefficient):
    u_gt = helmholtz_2d_exact_u(y, x, a1, a2)
    u_yy = -(a1*torch.pi)**2 * u_gt
    u_xx = -(a2*torch.pi)**2 * u_gt
    return  u_yy + u_xx + coefficient*u_gt

def generate_Helmholtz_2d_inverse_data(a1, a2):
    # test points
    y = torch.linspace(-1, 1, 50) 
    x = torch.linspace(-1, 1, 50)
    y, x = torch.meshgrid([y, x], indexing='ij')
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    u_test = helmholtz_2d_exact_u(y_test, x_test, a1, a2)
    return y_test, x_test, u_test    

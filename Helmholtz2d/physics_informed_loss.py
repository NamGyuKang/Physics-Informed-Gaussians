import torch

def Helmholtz_2d(u, y, x, coefficient, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), True, True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), True, True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
                    
    if problem == 'forward':
        f =  u_yy + u_xx + coefficient*u
    elif problem == 'inverse':
        f =  u_yy + u_xx + inverse_lambda*u
    return f

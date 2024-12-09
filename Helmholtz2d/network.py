import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        # parameters
        self.num_layers = args.num_layers
        
        # using cell
        self.interp = args.interp
        self.problem = args.problem
        self.mlp_dim = args.mlp_dim
        self.num_gaussians = args.num_gaussians
        self.sigma_init = args.sigma_init

        if self.problem == 'inverse':
            self.lambda_1 = torch.nn.Parameter(torch.zeros(1))
        # Network dimension
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.in_dim = args.in_dim

        if self.in_dim == 2:
            self.mu_t = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
            self.mu_x = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
            self.mu_t.data.uniform_(-1., 1.)
            self.mu_x.data.uniform_(-1., 1.)
            self.mu_t.requires_grad = True
            self.mu_x.requires_grad = True
            self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.mlp_dim, self.num_gaussians)))
            self.sigma_t = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
            self.sigma_x = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
            self.weight.requires_grad = True
            self.sigma_t.requires_grad = True
            self.sigma_x.requires_grad = True
            # if args.full_cov:
            #     self.r1 = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians)))
            #     self.r2 = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians)) * 0.0)

        elif self.in_dim == 3:
                self.mu_t = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_x = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_y = torch.nn.Parameter(torch.rand(self.mlp_dim, self.num_gaussians))
                self.mu_t.data.uniform_(-1, 1)
                self.mu_x.data.uniform_(-1, 1)
                self.mu_y.data.uniform_(-1, 1)
                self.mu_t.requires_grad = True
                self.mu_x.requires_grad = True
                self.mu_y.requires_grad = True
                self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.mlp_dim, self.num_gaussians)))
                self.sigma_t = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_x = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.sigma_y = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians))*self.sigma_init)
                self.weight.requires_grad = True
                self.sigma_t.requires_grad = True
                self.sigma_x.requires_grad = True
                self.sigma_y.requires_grad = True
            
        if args.activation=='relu':
            self.activation_fn = nn.ReLU()
        elif args.activation=='leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        elif args.activation=='sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif args.activation=='softplus':
            self.activation_fn = nn.Softplus()
        elif args.activation=='tanh':
            self.activation_fn = nn.Tanh()
        elif args.activation=='gelu':
            self.activation_fn = nn.GELU()
        elif args.activation =='logsigmoid':
            self.activation_fn = nn.LogSigmoid()
        elif args.activation =='hardsigmoid':
            self.activation_fn = nn.Hardsigmoid()
        elif args.activation =='elu':
            self.activation_fn = nn.ELU()
        elif args.activation =='celu':
            self.activation_fn = nn.CELU()            
        elif args.activation =='selu':
            self.activation_fn = nn.SELU() 
        elif args.activation =='silu':
            self.activation_fn = nn.SiLU()  
        elif args.activation == 'sin':
            self.activation_fn = Sin()
        else:
            raise NotImplementedError
      
        if self.num_layers==0:
            return
        
        ''' see the Section "Neural network and Grid Representations" in the paper.
                    we built the Neural network. '''
        self.net = []
        input_dim = self.mlp_dim
        if self.num_layers < 2:
            self.net.append(self.activation_fn)
            self.net.append(torch.nn.Linear(input_dim, self.out_dim))
        else:
            self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
            self.net.append(self.activation_fn)
            for i in range(self.num_layers-2): 
                self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                self.net.append(self.activation_fn)
            self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
        
        # deploy layers
        self.net = nn.Sequential(*self.net)

    def additional_params(self):
        self.r1 = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians)))
        self.r2 = torch.nn.Parameter(torch.ones((self.mlp_dim, self.num_gaussians)) * 0.0)

    def forward(self, x):
        if self.in_dim==2:
            means = torch.stack([self.mu_t, self.mu_x], -1)
            sigmas = torch.stack([self.sigma_t, self.sigma_x], -1)
 
            if self.args.full_cov:
                L = self.build_scaling_rotation_2x2(sigmas, self.r1, self.r2)
                cov = (L @ L.transpose(2, 3)).unsqueeze(1)
                d = x.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
                out = cov @ d[..., None]
                out = d[..., None, :] @ out
                feats = (torch.exp(-0.5*out.squeeze()) * self.weight.unsqueeze(1)).sum(-1).t()

            else:
                x = x.unsqueeze(0).repeat(self.mlp_dim, 1, 1)
                feats = self.gaussian_sample(x, means, sigmas, self.weight)
            
        elif self.in_dim ==3:
            means = torch.stack([self.mu_t, self.mu_x, self.mu_y], -1)
            sigmas = torch.stack([self.sigma_t, self.sigma_x, self.sigma_y], -1)
            x = x.unsqueeze(0).repeat(self.mlp_dim, 1, 1)
            feats = self.gaussian_sample(x, means, sigmas, self.weight)
            
        if self.num_layers > 0:
            out = self.net(feats)        
        else:
            out = feats.mean(0).squeeze().view(-1, 1)
            
        return out
    
    def gaussian_sample(self, X, means, sigmas, weight):
        means = means.unsqueeze(1)   # (k, 1, g, d)
        sigmas = sigmas.unsqueeze(1) # (k, 1, g, d)
        weight = weight.squeeze().unsqueeze(1) # (k, 1, g)
        X = X.unsqueeze(2)           # (1, p, 1, d)

        exponent = (((X - means)/sigmas)**2).sum(-1)
        gaussians = torch.exp(-0.5*exponent) * weight
        output = gaussians.sum(-1).t()

        return output

    def build_rotation_2x2(self, r):
        norm = torch.sqrt(r[..., 0]**2 + r[..., 1]**2)
        q = r / norm[..., None]

        # Define elements for the 2x2 matrix based on the normalized vector
        r, x = q[..., 0], q[..., 1]

        # Create a 2x2 matrix
        R = torch.zeros((*q.shape[:-1], 2, 2), dtype=r.dtype, device=r.device)
        R[..., 0, 0] = r
        R[..., 0, 1] = -x
        R[..., 1, 0] = x
        R[..., 1, 1] = r

        return R

    def build_scaling_rotation_2x2(self, s, r1, r2):
        r = torch.stack([r1, r2], -1)
        R = self.build_rotation_2x2(r)
        # Create a 2x2 scaling matrix
        L = torch.zeros((s.shape[0], s.shape[1], 2, 2), dtype=s.dtype, device=s.device)
        L[..., 0, 0] = 1.0 / s[..., 0]
        L[..., 1, 1] = 1.0 / s[..., 1]
        
        # Apply the 2x2 rotation to the scaling matrix
        L = R @ L
        
        return L
    
import numpy as np
import torch
import torch.optim as optim
import os
import sys
from matplotlib import rc
from tqdm import tqdm
from test_pde import *
from data_generator import *
from physics_informed_loss import *
from ground_truth import *

rc('text', usetex=False)

def pde_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag):
    if pde == '2d_helmholtz':
        num_test = 250
        Helmholtz_2d_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag, num_test)

class PIG():
    def __init__(self, network, args, PDE):
        self.args = args
        # deep neural networks
        self.dnn = network
        # random sampling at every iteration
        self.random_f = args.random_f
        # number of points
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.num_ic = args.num_init
        self.num_bc = args.num_init
        self.output_path = "results/figures/{}".format(args.tag)
        self.tag = args.tag
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.f_scale = args.f_scale
        self.u_scale = args.u_scale
        
        # TV regularization coefficient
        self.lamb = args.lamb
        
        
        self.use_cell = args.use_cell
        
        # optimizers: using the same settings
        self.optim = args.optim
        self.lr = args.lr
        self.max_iter = args.max_iter
        self.set_optimizer()
        
        self.iter = 0
        self.loss_list = []
        self.loss_b = 0
        self.loss_tv = 0

        self.exist_pde_source_term = False
        self.boundary_condition = False
        self.number_of_boundary = 0
        self.boundary_gradient_condition = False
        self.mixed_boundary_condition = False

        self.pde = PDE
        
        if self.pde == '2d_helmholtz':
            ''' PDE attribution '''
            self.exist_pde_source_term = True
            self.boundary_condition = True
            self.number_of_boundary = 1
            self.boundary_gradient_condition = False
            self.mixed_boundary_condition = False
            ''' PDE parameter '''
            self.a1 = args.a1
            self.a2 = args.a2
            self.coefficient = args.lambda_1
            ''' load data '''
            self.t_test, self.x_test, self.u_test = generate_Helmholtz_2d_test_data(self.num_test, self.a1, self.a2)
            self.t_train_f, self.x_train_f, self.u_train_f, self.t_train, self.x_train, self.u_train = generate_Helmholtz_2d_train_data(self.num_train, self.num_bc, self.a1, self.a2, self.coefficient)
            if self.args.problem == 'inverse':
                self.t_inverse_data, self.x_inverse_data, self.u_inverse_data = generate_Helmholtz_2d_inverse_data(self.a1, self.a2)

    def set_optimizer(self):
        if self.optim == 'lbfgs':
            self.optimizer = optim.LBFGS(
                self.dnn.parameters(), 
                lr=self.lr, 
                #max_iter=self.max_iter, 
                #max_eval=50000,
                #history_size=50,
                #tolerance_grad=1e-6,
                #tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )
        elif self.optim == 'adam':
            self.optimizer = optim.Adam(self.dnn.parameters(), lr = self.lr) #, capturable=True
            self.optimizer.param_groups[0]['capturable'] = True
            gamma = 0.9995
            # gamma = 0.95
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            raise NotImplementedError()

    def net_u_2d(self, t, x, pde):
        # if self.use_cell:
        ''' normalize to [-1, 1] '''
        x = torch.cat([t, x], dim=1)
        u = self.dnn(x)
        
        return u
 
    def net_f_2d(self, t, x, pde):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u_2d(t, x, pde) # , u2, mx,my,sx,sy, weight
        
        if self.args.problem == 'forward':
            lambda_1 = None
        elif self.args.problem == 'inverse':
            lambda_1 = self.dnn.lambda_1
        
        if pde == '2d_helmholtz': # , u2, mx,my,sx,sy, weight,
            f = Helmholtz_2d(u, t, x, self.coefficient, lambda_1, self.args.problem)

        return f
    
    def tv(self):
        return self.dnn.tv()


    def loss_func_2d(self):
        self.optimizer.zero_grad()
        f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde)
        loss_f = torch.mean(f_pred ** 2)
        if self.exist_pde_source_term:
            loss_f = torch.mean((self.u_train_f - f_pred)**2)

        if self.args.problem == 'forward':
            ''' initial condition '''
            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            
            if self.boundary_condition:
                ''' boundary condition '''
                if self.number_of_boundary == 1:
                    u_bc1_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
                    loss_b = torch.mean((self.u_train - u_bc1_pred) ** 2)
                elif self.number_of_boundary ==2:            
                    u_bc1_pred = self.net_u_2d(self.t_bc1_train, self.x_bc1_train, self.pde)
                    u_bc2_pred = self.net_u_2d(self.t_bc2_train, self.x_bc2_train, self.pde)
                    loss_b = torch.mean((u_bc1_pred - u_bc2_pred) ** 2)

                if self.boundary_gradient_condition:
                    ''' boundary gradient condition '''
                    u_bc1_x = torch.autograd.grad(u_bc1_pred, self.x_bc1_train, grad_outputs=torch.ones_like(u_bc1_pred), retain_graph=True, create_graph=True)[0]
                    u_bc2_x = torch.autograd.grad(u_bc2_pred, self.x_bc2_train, grad_outputs=torch.ones_like(u_bc2_pred), retain_graph=True, create_graph=True)[0]
                    loss_b = torch.mean((u_bc1_x - u_bc2_x) ** 2)
                    
                    if self.mixed_boundary_condition:
                        ''' Summation (boundary condition, 1st order boundary gradient condition) '''
                        if self.use_b_loss:
                            loss_b = torch.mean((u_bc1_pred - u_bc2_pred)**2) + torch.mean((u_bc1_x - u_bc2_x)**2)
                        else:
                            loss_b = torch.zeros(1)

            
            if self.pde == '2d_helmholtz':
                scaled_loss = loss_u + self.f_scale*loss_f + self.lamb*self.loss_tv
            
        elif self.args.problem == 'inverse':
            u_data_pred = self.net_u_2d(self.t_inverse_data, self.x_inverse_data, self.pde)
            loss_data = torch.mean((self.u_inverse_data - u_data_pred)**2)
            scaled_loss = loss_data + self.f_scale*loss_f
        
        scaled_loss.backward(retain_graph = True)
        # for loggin purpose

        self.loss_f = loss_f.item()
        if self.args.problem == 'forward':
            self.loss_u = loss_u.item()
        elif self.args.problem == 'inverse':
            self.loss_data = loss_data.item()

        return scaled_loss

    def train(self):
        # Backward and optimize
        for it in tqdm(range(self.max_iter)):
            self.dnn.train()
            self.it = it
            
            if self.args.full_cov:
                if it == 0:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.dnn.to(device)
                    checkpoint = torch.load('your_diagonal_covariance_pretrain_model')
                    self.dnn.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    optimizer_state = checkpoint.get('optimizer_state_dict', None)
                    if optimizer_state:
                        for state in optimizer_state['state'].values():
                            if 'step' in state:
                                state['step'] = state['step'].to(device)
                    self.optimizer.load_state_dict(optimizer_state)
                    self.dnn.additional_params()
                    new_params = [p for p in self.dnn.parameters()]
                    self.optimizer.add_param_group({'params': new_params[-6:-4]})
            
            if self.optim == 'lbfgs':
                self.optimizer.step(self.loss_func_2d)
                if self.args.problem == 'forward':
                    if it % 15 ==0:
                        print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_b: %.5e, Loss_f: %.5e, Loss_tv: %.5e'%(
                            it+1, self.loss_u+self.loss_b+self.loss_f, self.loss_u, self.loss_b, self.loss_f, self.loss_tv))
                elif self.args.problem == 'inverse':
                    if it % 1 ==0:
                        if self.pde[:2] != '3d':
                            print('Iter %d, lambda: %.5e, Loss: %.5e, Loss_data: %.5e, Loss_f: %.5e, Loss_tv: %.5e'%(
                                it+1, self.dnn.lambda_1, self.loss_data+self.loss_f, self.loss_data, self.loss_f, self.loss_tv))
                        else:
                            print('Iter %d, lda1: %.5e, lda2: %.5e, Loss: %.5e, Loss_u: %.5e, Loss_v: %.5e, Loss_f_u: %.5e, Loss_f_v: %.5e'%(
                                it+1, self.dnn.lambda_1.item(), self.dnn.lambda_2.item(), self.loss, self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v))
                sys.stdout.flush()
            else:
                if self.args.problem == 'forward':
                    self.optimizer.zero_grad()
                    ''' Adam optimizer for 2d PDEs '''
                    f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde)
                    loss_f = torch.mean(f_pred ** 2)
                    if self.exist_pde_source_term:
                        loss_f = torch.mean((self.u_train_f - f_pred)**2)

                    if self.args.problem == 'forward':
                        ''' initial condition '''
                        if self.args.pde != '2d_helmholtz':
                            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
                            loss_u = torch.mean((self.u_train - u_pred) ** 2)
                        
                        if self.boundary_condition:
                            ''' boundary condition '''
                            if self.number_of_boundary == 1:
                                u_bc1_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
                                loss_b = torch.mean((self.u_train - u_bc1_pred) ** 2)
                            elif self.number_of_boundary ==2:            
                                u_bc1_pred = self.net_u_2d(self.t_bc1_train, self.x_bc1_train, self.pde)
                                u_bc2_pred = self.net_u_2d(self.t_bc2_train, self.x_bc2_train, self.pde)
                                loss_b = torch.mean((u_bc1_pred - u_bc2_pred) ** 2)

                            if self.boundary_gradient_condition:
                                ''' boundary gradient condition '''
                                u_bc1_x = torch.autograd.grad(u_bc1_pred, self.x_bc1_train, grad_outputs=torch.ones_like(u_bc1_pred), retain_graph=True, create_graph=True)[0]
                                u_bc2_x = torch.autograd.grad(u_bc2_pred, self.x_bc2_train, grad_outputs=torch.ones_like(u_bc2_pred), retain_graph=True, create_graph=True)[0]
                                loss_b = torch.mean((u_bc1_x - u_bc2_x) ** 2)
                                
                                if self.mixed_boundary_condition:
                                    ''' Summation (boundary condition, 1st order boundary gradient condition) '''
                                    if self.use_b_loss:
                                        loss_b = torch.mean((u_bc1_pred - u_bc2_pred)**2) + torch.mean((u_bc1_x - u_bc2_x)**2)
                                    else:
                                        loss_b = torch.zeros(1)

                        if self.pde == '2d_helmholtz':
                            scaled_loss = loss_b + self.f_scale*loss_f + self.lamb*self.loss_tv
                        
                        
                elif self.args.problem == 'inverse':
                    f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde)
                    loss_f = torch.mean(f_pred ** 2)
                    if self.exist_pde_source_term:
                        loss_f = torch.mean((self.u_train_f - f_pred)**2)
                    u_data_pred = self.net_u_2d(self.t_inverse_data, self.x_inverse_data, self.pde)

                scaled_loss.backward()
                # for loggin purpose
                self.optimizer.step()
                # if it % 100 == 0 and it > 0:
                self.scheduler.step()

                self.loss_f = loss_f.item()
                if self.args.problem == 'forward':
                    self.loss_b = loss_b.item()
                elif self.args.problem == 'inverse':
                    self.loss_data = loss_data.item()
                    
                if it % 100 == 0:
                    print('Iter %d, Loss: %.5e, Loss_b: %.5e, Loss_f: %.5e' % (
                            it, self.loss_b + self.loss_f, self.loss_b, self.loss_f))
                    sys.stdout.flush()

            if it % 1== 0:
                self.test(it, self.pde)

            self.iter += 1

            if self.args.pretrain:
                if it % 50 == 0:
                    checkpoint = {
                        'model_state_dict': self.dnn.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint, self.output_path + '/model_checkpoint_it_{}.pth'.format(it))

            # Every interation, we randomly sample collocation points
            if self.random_f:
                if self.pde == '2d_helmholtz':
                    self.t_train_f, self.x_train_f, self.u_train_f, self.t_train, self.x_train, self.u_train = generate_Helmholtz_2d_train_data(self.num_train, self.num_bc, self.a1, self.a2, self.coefficient)
                

    def test(self, it, pde):
        self.dnn.eval()
        if self.args.problem == 'forward':
            lambda_1 = None
        elif self.args.problem == 'inverse':
            lambda_1 = self.dnn.lambda_1
        pde_test(pde, self.t_test, self.x_test, self.u_test, lambda_1, self.net_u_2d, self.args.problem, it, self.loss_list, self.output_path, self.tag)
    
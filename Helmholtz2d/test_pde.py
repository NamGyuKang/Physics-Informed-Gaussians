import numpy as np
import torch
import sys
from plot import *

def save_loss_list(problem, loss_list, it, output_path, save_it = 50):
    if problem =='inverse':
        save_it = 5
    if it % save_it ==0:
        np.save(output_path + "/loss_{}.png".format(it), loss_list)

def Helmholtz_2d_test(pde, y_test, x_test, u_test, inverse_lambda, net_u, problem, it, loss_list, output_path, tag, num_test):
    u_pred = net_u(y_test, x_test, pde)
    u_pred_arr = u_pred.detach().cpu().numpy()
    u_test_arr = u_test.detach().cpu().numpy()
    
    l2_loss = np.linalg.norm(u_pred_arr - u_test_arr) / np.linalg.norm(u_test_arr)
    if problem == 'forward':
        loss_list.append(l2_loss)
        if it % 100 ==0 :
            print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
            # logger.error('Iter %d, l2_Loss: %.5e', it+1, l2_loss)
            
    elif problem == 'inverse':
        print('[Test Iter:%d, lambda: %.5e, Loss: %.5e]'%(it, inverse_lambda, l2_loss))
        # logger.error('Iter %d, lambda: %.5e, l2_Loss: %.5e', it+1, lambda_1, l2_loss)   
        loss_list.append(inverse_lambda.item())
    
    sys.stdout.flush()
    save_loss_list(problem, loss_list, it, output_path)

    if it % 1 == 0 :
        Helmholtz_2d_plot(it, y_test, x_test, u_pred.detach(), u_test, num_test, output_path, tag)
        
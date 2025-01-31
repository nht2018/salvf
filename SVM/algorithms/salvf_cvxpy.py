import cvxpy as cp
import numpy as np
import time

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split


def salvf(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs, verbose=True):

    feature=x_train.shape[1] # = 8
    ######### parameters
    c_array = torch.Tensor(y_train.shape[0]).uniform_(-3,3)

    ###### salvf paramter
    alpha = hparams['alpha'] # step size of outer loop
    c1 = hparams['c1']         # penalty paramether
    c2 = hparams['c2']         # penalty paramether

    C = cp.Parameter(y_train.shape[0], nonneg=True)
    
    # Parameters for eq. (12)
    w = cp.Variable(feature)
    b = cp.Variable()
    xi = cp.Variable(y_train.shape[0], nonneg=True)
    

    # Parameters for eq. (13)
    w_F = cp.Variable(feature)
    b_F = cp.Variable()
    xi_F = cp.Variable(y_train.shape[0], nonneg=True)

    ######### 2 level objectives
    loss_lower =  0.5*cp.norm(w, 2)**2 + 0.5 * (cp.scalar_product(C, cp.power(xi,2))) # cp.exp(C)
        
    # Compute the final expression
    

    # Create two constraints.
    constraints=[]
    constraints_value=[]
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) <= 0)
        constraints_value.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, x_train[i])+b) )
    
    # constraints_xi = [xi <= C]

    constraints_F=[]
    constraints_value_F=[]
    for i in range(y_train.shape[0]):
        constraints_F.append(1 - xi_F[i] - y_train[i] * (cp.scalar_product(w_F, x_train[i])+b_F) <= 0)
        constraints_value_F.append(1 - xi_F[i] - y_train[i] * (cp.scalar_product(w_F, x_train[i])+b_F) )

    # constraints_xi_F = [xi_F <= C]

    # Form objective.
    obj_lower = cp.Minimize(loss_lower)


    # Form and solve problem.
    prob_lower = cp.Problem(obj_lower, constraints)

    w_tensor = torch.ones(1,feature)
    b_tensor = torch.tensor(0.)
    xi_tensor = torch.tensor(y_train.shape[0])
    w_F_tensor = w_tensor.clone()
    b_F_tensor = b_tensor.clone()
    xi_F_tensor = xi_tensor.clone()

    C_tensor=torch.ones(1,feature).requires_grad_()

    
    # For storage
    val_loss_list=[]
    test_loss_list=[]
    val_acc_list=[]
    test_acc_list=[]
    time_computation=[]
    algorithm_start_time=time.time()

    metrics = []
    variables = []

    for epoch in range(epochs):

        variables.append({
            'C': C_tensor,
            'xi': xi_tensor,
            'w': w_tensor,
            'b': b_tensor,
            'xi_F': xi_F_tensor,
            'w_F': w_F_tensor,
            'b_F': b_F_tensor
        })

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_F_tensor, b_F_tensor) # / torch.linalg.norm(w_tensor)

        x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
        x1 = x1 * F.linear(torch.Tensor(x_test), w_F_tensor, b_F_tensor) # / torch.linalg.norm(w_tensor)
        # test_loss_upper= torch.sum(torch.sigmoid(x1))
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss_F = (torch.sum(torch.exp(1-x))).detach().numpy()/y_val.shape[0]
        test_loss_F = test_loss_upper.detach().numpy()/y_test.shape[0]

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_val), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)

        x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
        x1 = x1 * F.linear(torch.Tensor(x_test), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        # test_loss_upper= torch.sum(torch.sigmoid(x1))
        test_loss_upper= torch.sum(torch.exp(1-x1))

        val_loss = (torch.sum(torch.exp(1-x))).detach().numpy()/y_val.shape[0]
        test_loss = test_loss_upper.detach().numpy()/y_test.shape[0]

        loss_upper = val_loss
        loss_lower = (1/2) * torch.sum(w_tensor**2)
        ###### Accuracy
        q = torch.tensor(y_train) * (w_tensor @ x_train.T + b_tensor)
        train_acc = (q>0).sum() / len(y_train)

        q = torch.tensor(y_val) * (w_tensor @ x_val.T + b_tensor)
        val_acc = (q>0).sum() / len(y_val)

        q = torch.tensor(y_test) * (w_tensor @ x_test.T + b_tensor)
        test_acc = (q>0).sum() / len(y_test)

        q = torch.tensor(y_train) * (w_F_tensor.detach() @ x_train.T + b_F_tensor)
        train_acc_F = (q>0).sum() / len(y_train)

        q = torch.tensor(y_val) * (w_F_tensor.detach() @ x_val.T + b_F_tensor)
        val_acc_F = (q>0).sum() / len(y_val)

        q = torch.tensor(y_test) * (w_F_tensor.detach() @ x_test.T + b_F_tensor)
        test_acc_F = (q>0).sum() / len(y_test)

        metrics.append({
            # 'train_loss': train_loss,
            'train_acc': train_acc,
            'train_acc_F': train_acc_F,
            'val_loss': val_loss,
            'val_loss_F': val_loss_F,
            'val_acc': val_acc,
            'val_acc_F': val_acc_F,
            'test_loss': test_loss,
            'test_loss_F': test_loss_F,
            'test_acc': test_acc,
            'test_acc_F': test_acc_F,
            'loss_upper': loss_upper,
            'loss_lower': prob_lower.value,
            'time_computation': time.time()-algorithm_start_time
        })

        C_array_tensor = torch.exp(c_array)
        # print(c_array_value_np.sum(),sum(c_array))
        # print((c_array_value_np))
        C.value= C_array_tensor.detach().numpy() / C_array_tensor.detach().numpy().sum()

        ###### Solve Eq.(12), (13)
        prob_lower.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)  
       
        # print("time: ",end-begin)

        # dual_variables = np.array([ constraints[i].dual_value for i in range(len(constraints))])
        # constraints_value_1= np.array([ constraints_value[i].value for i in range(len(constraints))])
        # dual_variables_xi = constraints_xi[0].dual_value
        
        # dual_variables_F = np.array([ constraints_F[i].dual_value for i in range(len(constraints_F))])
        # constraints_value_1_F= np.array([ constraints_value_F[i].value for i in range(len(constraints_F))])
        # dual_variables_xi_F = constraints_xi_F[0].dual_value

        ############# Calculate gradient
        # Before backward pass, zero out the gradients for each parameter

                
        w_tensor=torch.Tensor(np.array([w.value])).requires_grad_(False) #.requires_grad_()
        b_tensor=torch.Tensor(np.array([b.value])).requires_grad_(False) #.requires_grad_()
        xi_tensor =torch.Tensor(np.array([xi.value])).requires_grad_(False)

        C_tensor = torch.Tensor(np.array(C.value)).requires_grad_()


        if epoch == 0:
            w_F_tensor=torch.Tensor(np.array([w.value])).requires_grad_()
            b_F_tensor=torch.Tensor(np.array([b.value])).requires_grad_()
            xi_F_tensor =torch.Tensor(np.array([xi.value])).requires_grad_()


        C_tensor.grad = None
        w_F_tensor.grad = None
        b_F_tensor.grad = None
        xi_F_tensor.grad = None

        loss_lower =  0.5*torch.linalg.norm(w_tensor) + 0.5 * C_tensor@ (xi_tensor**2).T
        loss_lower.backward()

        loss_upper = 0.5*torch.linalg.norm(w_F_tensor) + 0.5 * C_tensor @ (xi_F_tensor**2).T 
        x = torch.reshape(torch.Tensor(y_train), (torch.Tensor(y_train).shape[0],1)) 
        x = x* F.linear(torch.Tensor(x_train), w_F_tensor, b_F_tensor) # / torch.linalg.norm(w_tensor)
        # print(torch.sum( torch.maximum(1 - xi_F_tensor - x , torch.tensor(0.))** 2 ) / y_train.shape[0])
        loss_upper += c2 * 0.5 * torch.sum( torch.maximum(1 - xi_F_tensor - x , torch.tensor(0., requires_grad=True))** 2 ) / y_train.shape[0]
  
        loss_upper.backward()
        if epoch > 0:
            # gradient type update
            
            w_F_tensor.data.add_(w_F_tensor.grad, alpha = -alpha)
            if b_F_tensor.grad is not None:
                b_F_tensor.data.add_(b_F_tensor.grad, alpha = -alpha)
            xi_F_tensor.data.add_(xi_F_tensor.grad, alpha=-alpha)  # Update xi_F_tensor using its gradient
            xi_F_tensor.data = torch.maximum(xi_F_tensor.data, torch.tensor(0.))  # Clamp to a minimum of 0
            xi_F_tensor.data = torch.minimum(xi_F_tensor.data, torch.tensor(1.))  # Clamp to a maximum of 1
       

        # x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) 
        # x = x* F.linear(torch.Tensor(x_val), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        # # loss_upper= torch.sum(torch.sigmoid(x))
        # loss_upper= torch.sum(torch.exp(1-x))
        
        # x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) 
        # x1 = x1 * F.linear(torch.Tensor(x_test), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        # # test_loss_upper= torch.sum(torch.sigmoid(x1))
        # test_loss_upper= torch.sum(torch.exp(1-x1))

        # val_loss = loss_upper.detach().numpy()/150
        # test_loss = test_loss_upper.detach().numpy()/118

        ############# update on upper level variable C
        C_tensor.data -= alpha* c1 *(xi_F_tensor[0]**2-xi_tensor[0]**2)#grads_C_g#gam*(grads_C_g_F-)
        C_tensor = torch.maximum(C_tensor, torch.tensor(1e-4))
        
        #################


        if epoch%20==0 and verbose:
            print(f"Epoch [{epoch}/{epochs}]:",
              "val acc: {:.2f}".format(val_acc),
              "val loss: {:.2f}".format(val_loss),
              "test acc: {:.2f}".format(test_acc),
              "test loss: {:.2f}".format(test_loss))
            # print(f"Epoch [{j}/{epoch}]:","upper_loss: ", loss_upper.detach().numpy()/15.0, "test_loss_upper: ", test_loss_upper.detach().numpy()/11.8)

        val_loss_list.append(val_loss) # length 150
        test_loss_list.append(test_loss) # length 118
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        time_computation.append(time.time()-algorithm_start_time)




    return metrics, variables


if __name__ == "__main__":
    ############ Load data code ###########

    data_utils = load_diabetes()

    data_list=[]

    f = open("../diabete.txt",encoding = "utf-8")
    a_list=f.readlines()
    f.close()
    for line in a_list:
        line1=line.replace('\n', '')
        line2=list(line1.split(' '))
        y=float(line2[0])
        x= [float(line2[i].split(':')[1]) for i in (1,2,3,4,5,6,7,8)]
        data_list.append(x+[y])


    data_array_1=np.array(data_list)[:,:-1]
    data_array_0=np.ones((data_array_1.shape[0],1))
    data_array_2=data_array_1*data_array_1
    data_array_3=np.empty((data_array_1.shape[0],0))

    for i in range(data_array_1.shape[1]):
        for j in range(data_array_1.shape[1]):
            if i<j:
                data_array_i=data_array_1[:,i]*data_array_1[:,j]
                data_array_i=np.reshape(data_array_i,(-1,1))
                data_array_3=np.hstack((data_array_3,data_array_i))

    data_array_4=np.reshape(np.array(data_list)[:,-1],(-1,1))
    data=np.hstack((data_array_0,data_array_1,data_array_2,data_array_3,data_array_4))

    n_train = 500
    n_val = 150

    metrics = []
    variables = []

    hparams = {
        'gam': 5,
        'alpha': 0.1
    }

    epochs = 80
    plot_results = True

    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)

        metrics_seed, variables_seed = salvf(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
        metrics.append(metrics_seed)
        variables_seed.append(variables_seed)

    train_acc = np.array([[x['train_acc'] for x in metrics] for metrics in metrics])
    val_acc = np.array([[x['val_acc'] for x in metrics] for metrics in metrics])
    test_acc = np.array([[x['test_acc'] for x in metrics] for metrics in metrics])

    val_loss = np.array([[x['val_loss'] for x in metrics] for metrics in metrics])
    test_loss = np.array([[x['test_loss'] for x in metrics] for metrics in metrics])

    time_computation = np.array([[x['time_computation'] for x in metrics] for metrics in metrics])

    if plot_results:
        val_loss_mean=np.mean(val_loss,axis=0)
        val_loss_sd=np.std(val_loss,axis=0)/2.0
        test_loss_mean=np.mean(test_loss,axis=0)
        test_loss_sd=np.std(test_loss,axis=0)/2.0

        val_acc_mean=np.mean(val_acc,axis=0)
        val_acc_sd=np.std(val_acc,axis=0)/2.0
        test_acc_mean=np.mean(test_acc,axis=0)
        test_acc_sd=np.std(test_acc,axis=0)/2.0

        axis = np.mean(time_computation,axis=0)

        plt.rcParams.update({'font.size': 18})
        plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
        plt.rcParams['axes.unicode_minus']=False #显示负号
        axis=time_computation.mean(0)
        plt.figure(figsize=(8,6))
        #plt.grid(linestyle = "--") #设置背景网格线为虚线
        ax = plt.gca()
        plt.plot(axis,val_loss_mean,'-',label="Training loss")
        ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
        plt.plot(axis,test_loss_mean,'--',label="Test loss")
        ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        #plt.legend(loc=4)
        plt.ylabel("Loss")
        #plt.xlim(-0.5,3.5)
        #plt.ylim(0.5,1.0)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
        plt.savefig('ho_svm_kernel_1.pdf') 
        #plt.show()

        plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.plot(axis,val_acc_mean,'-',label="Training accuracy")
        ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
        plt.plot(axis,test_acc_mean,'--',label="Test accuracy")
        ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        plt.ylabel("Accuracy")
        # plt.ylim(0.64,0.8)
        #plt.legend(loc=4)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
        plt.savefig('ho_svm_kernel_2.pdf') 
        plt.show()

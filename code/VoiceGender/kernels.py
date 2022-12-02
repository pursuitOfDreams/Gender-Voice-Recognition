import torch

# Hyperparameters
sigma_gaussian = 1.52
gamma_rbf = 0.11
sigma_laplace = 12.75
constant_poly = 1
degree_poly = 2

def gaussian_kernel(a,b):
    a_2 = torch.diag(torch.matmul(a,a.T))
    b_2 = torch.diag(torch.matmul(b,b.T))
    ab = torch.matmul(a,b.T)
    return torch.exp(-(a_2[:,None]+b_2[None,:]-2*ab)/(2*sigma_gaussian*sigma_gaussian))

def rbf_kernel(a,b):
    a_2 = torch.diag(torch.matmul(a,a.T))
    b_2 = torch.diag(torch.matmul(b,b.T))
    ab = torch.matmul(a,b.T)
    return torch.exp((-1)*gamma_rbf*(a_2[:,None]+b_2[None,:]-2*ab))

def laplace_kernel(a,b):
    diff = a[:,None,:] - b 
    l1_norm = torch.linalg.norm(diff, ord=1, dim=2)
    return torch.exp(-l1_norm/sigma_laplace)

def poly_kernel(X, Y):
   K = torch.zeros((X.shape[0],Y.shape[0]))
#    K = (torch.matmul(X,Y.T) + constant_poly)**degree_poly
   K = torch.pow(torch.matmul(X,Y.T) + constant_poly,degree_poly)

   return K
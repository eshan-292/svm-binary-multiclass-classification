import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()


def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    # assert X.ndim == 2
    no = kwargs.get("no", 1)
    if no ==1:
      kernel_matrix = X @ X.T     #np.dot(X, X.T)
      return kernel_matrix
    elif no==2:
      Y = kwargs.get("Y")
      return np.dot(X, Y)



# Polynomial kernel
def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    # assert X.ndim == 2
    no = kwargs.get("no", 1)
    degree = kwargs.get("degree", 2)
    gamma = kwargs.get("gamma", 1)
    coef0 = kwargs.get("coef0", 0)
    if no ==1:
      
      return (gamma * (X @ X.T) + coef0) ** degree
    elif no==2:
      Y = kwargs.get("Y")

      return (gamma * (np.dot(X,Y)) + coef0) ** degree

    
    
    

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    # assert X.ndim == 2
    no = kwargs.get("no", 1)
    gamma = kwargs.get("gamma", 1)
    
    if no ==1:
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                K[i][j] = np.exp(-gamma * dist)
                K[j][i] = K[i][j]
        return K
    elif no==2:
        Y = kwargs.get("Y")
        n_samples = X.shape[0]
        m_samples = Y.shape[0]
        
        #Print the number of samples
        print ("No of samples in X: ", n_samples)
        print ("No of samples in Y: ", m_samples)

        K = np.zeros((n_samples, m_samples))
        for i in range(n_samples):
            for j in range(m_samples):
                # print("HEY")

                dist = np.linalg.norm(X[i] - Y[j])
                K[i][j] = np.exp(-gamma * dist)
        return K
        

    

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    # assert X.ndim == 2
    gamma = kwargs.get("gamma", 1)
    coef0 = kwargs.get("coef0", 0)
    no = kwargs.get("no", 1)
    if no ==1:
        return np.tanh(gamma * (X @ X.T) + coef0)
    elif no==2:
        Y = kwargs.get("Y")
        return np.tanh(gamma * (np.dot(X,Y)) + coef0)
    
    
    

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    no = kwargs.get("no", 1)
    gamma = kwargs.get("gamma", 1)

    if no ==1:
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                K[i][j] = np.exp(-gamma * dist)
                K[j][i] = K[i][j]
        return K
    elif no==2:
        Y = kwargs.get("Y")
        n_samples = X.shape[0]
        m_samples = Y.shape[0]
        K = np.zeros((n_samples, m_samples))

        


        for i in range(n_samples):
            for j in range(m_samples):
                dist = np.linalg.norm(X[i] - Y[j])
                K[i][j] = np.exp(-gamma * dist)
        return K




def rbf_kernel2(X:np.ndarray, Y:np.ndarray, gamma)-> np.ndarray:
    # assert X.ndim == 2
    # assert Y.ndim == 2
    # gamma = kwargs.get("gamma", 1)
    n_samples = X.shape[0]
    m_samples = Y.shape[0]
    K = np.zeros((n_samples, m_samples))

    #Print the number of samples
    print ("No of samples in X: ", n_samples)
    print ("No of samples in Y: ", m_samples)
    for i in range(n_samples):
        for j in range(m_samples):
            dist = np.linalg.norm(X[i] - Y[j])
            K[i][j] = np.exp(-gamma * dist)
    return K
    


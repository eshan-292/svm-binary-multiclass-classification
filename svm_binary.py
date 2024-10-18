from typing import List
import numpy as np
import pandas as pd

from kernel import linear, polynomial, rbf, sigmoid, laplacian

import qpsolvers
import matplotlib.pyplot as plt

import seaborn as sns



    


class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        self.b = 0
        self.w = 0


        
    
    def fit(self, train_data_path:str)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors


        # Read the data
        data = pd.read_csv(train_data_path, header=None)
        
        # print(data.shape)
        # print(data)

        # Read X and y as numpy arrays
        

        X = data.iloc[1:, 1:-1].values
        # print(X)
        y = data.iloc[1:, -1].astype(float).values

        # Converting 0s to -1s
        y[y == 0] = -1

        # print(y.shape)
        print(y)
        
        
        # Compute the kernel matrix
        K = self.kernel(X, **self.kwargs)





        # print(K.shape)
        # print(K)

        # Compute the dual problem
        n_samples = X.shape[0]
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.diag(np.ones(n_samples) * -1)
        h = np.zeros(n_samples)
        A = y.reshape(1, -1)
        b = np.zeros(1)


        
        
        if self.C is not None:
            G = np.vstack((G, np.diag(np.ones(n_samples))))
            h = np.hstack((h, np.ones(n_samples) * self.C))
        
        alpha = qpsolvers.solve_qp(P, q, G, h, A, b, solver="cvxopt")
        

        print("Alpha Shape:- ", alpha.shape)
        print("Alpha:- ", alpha)
        

        # Compute the support vectors
        epsilon = 1e-7

        # selecting the indices of the support vectors having alpha > epsilon and alpha < C
        sv = ((alpha > epsilon) & (alpha < self.C))
        
        ind = np.arange(len(alpha))[sv]

        self.support_vectors = X[sv]
        # print("Support Vectors Shape:- ", self.support_vectors.shape)
        # print("Support Vectors:- ", self.support_vectors)
        self.alpha = alpha[sv]
        self.sv_y = y[sv]
        self.svind = ind


      

        print("Alpha Shape:- ", self.alpha.shape)
        print("Alpha:- ", self.alpha)
        # print(self.sv_y.shape)
        print(self.sv_y)
        
        # Compute the intercept
        
        self.b = 0
        for n in range(len(self.alpha)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alpha * self.sv_y * K[ind[n], sv])
            
        if len(self.alpha !=0):
          self.b /= len(self.alpha)

      


        print("Bias: " , self.b)
  

        # Compute the weights
        if self.kernel == linear:
          self.w = np.zeros(X.shape[1])
          for n in range(len(self.alpha)):
              self.w += self.alpha[n] * self.sv_y[n] * self.support_vectors[n]
        else:
          self.w = None
        
        
        # print(self.w.shape)
        # print("Weights: ", self.w)



    
        

    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        
        # Read the data
        data = pd.read_csv(test_data_path, header=None)
        X = data.iloc[1:, 1:].values
        # print(X.shape)
        
        
        # y = data.iloc[1:, -1].astype(float).values

        # Converting 0s to -1s
        # y[y == 0] = -1


        # Compute the kernel values between the test data and the support vectors
        # K = rbf_kernel2(X, self.support_vectors, 0.1)
        # K = np.transpose(K)
        #Print the shape of kernel
        # print(K.shape)

        #Print svind
        # print("Svind:- ", self.svind)
        #print alpha
        # print("Alpha:- ", self.alpha[39])
        #print sv_y
        # print("Sv_y:- ", self.sv_y[39])

        # Compute the predictions
        y_pred = np.zeros(len(X))

        print("length of X:- ", len(X))
        
        # for i in range(len(X)):
        #     s = 0
        #     for a, sv_y, sv in zip(self.alpha, self.sv_y, self.support_vectors):
        #         s += a * sv_y * rbf_kernel2(X[i], sv, 0.1)
        #     y_pred[i] = s
        # y_pred += self.b
        # y_pred = np.sign(y_pred)




        # return y_pred

        

        

        # print(y.shape)
        # print(y)

        if self.w is not None:
            y_pred = np.sign(np.dot(X, self.w) + self.b)
        
            # Replace all the -1s to 0s
            y_pred[y_pred == -1] = 0
            return y_pred

            
            
        else:
          y_predict = np.zeros(len(X))
          for i in range(len(X)):
              s = 0
              # print(len(zip(self.alpha, self.sv_y, self.support_vectors)))
              for a, sv_y, sv in zip(self.alpha, self.sv_y, self.support_vectors):
                
                # Adding kwargs
                kwargs = self.kwargs
                kwargs['no'] = 2
                kwargs['Y'] = sv


                s += a * sv_y * self.kernel(X[i], **kwargs)
                #   s += a * sv_y * rbf_kernel2(X[i], sv, 0.1 )
              y_predict[i] = s

          
          y_predict = np.sign(y_predict + self.b)

          # Replace all the -1s to 0s
          y_predict[y_predict == -1] = 0
            
          return y_predict
            # return np.sign(np.dot(self.alpha * self.sv_y, self.kernel(self.support_vectors, X, **self.kwargs)) + self.b)

    

    # Calculate the accuracy and F1 scores
    def score(self, test_data_path:str)->float:
        y_pred = self.predict(test_data_path)
        print(y_pred)
        y_true = pd.read_csv(test_data_path, header=None).iloc[1:, -1].astype(float).values

        # Converting 0s to -1s
        # y_true[y_true == 0] = -1

        print(y_true)
        


        # Calculating the confusion matrix
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        print("Confusion Matrix: ")
        print("True Positive: ", tp)
        print("False Positive: ", fp)
        print("False Negative: ", fn)
        print("True Negative: ", tn)

        #Plotting the confusion matrix using matplotlib 
        
        plt.figure(figsize=(10, 7))
        plt.title("Confusion Matrix")
        sns.heatmap([[tp, fp], [fn, tn]], annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        

        







        
        # Calculate the accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print("Accuracy: ", accuracy)

        # Calculate the F1 score
        # tp = np.sum((y_pred == 1) & (y_true == 1))
        # fp = np.sum((y_pred == 1) & (y_true == -1))
        # fn = np.sum((y_pred == -1) & (y_true == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print("F1 score: ", f1)

        return accuracy
    



































    # Multi-class SVM




    def multi_fit(self, X, y)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors


        # # Read the data
        # data = pd.read_csv(train_data_path, header=None)



        # # print(data.shape)
        # # print(data)

        # # Read X and y as numpy arrays
        

        # X = data.iloc[1:, 2:].values
        # # print(X)
        # y = data.iloc[1:, 1].astype(float).values

        # # Converting 0s to -1s
        # y[y == 0] = -1

        # # print(y.shape)
        # print(y)
        
        
        # Compute the kernel matrix
        K = self.kernel(X, **self.kwargs)
        # print(K.shape)
        # print(K)

        # Compute the dual problem
        n_samples = X.shape[0]
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.diag(np.ones(n_samples) * -1)
        h = np.zeros(n_samples)
        A = y.reshape(1, -1)
        b = np.zeros(1)


        
        
        if self.C is not None:
            G = np.vstack((G, np.diag(np.ones(n_samples))))
            h = np.hstack((h, np.ones(n_samples) * self.C))
        alpha = qpsolvers.solve_qp(P, q, G, h, A, b, solver="cvxopt")
        

        # print("Alpha Shape:- ", alpha.shape)
        print("Alpha:- ", alpha)
        

        # Compute the support vectors
        epsilon = 1e-6
        # sv = alpha > epsilon
        sv = ((alpha > epsilon) & (alpha < self.C))
        ind = np.arange(len(alpha))[sv]
        self.support_vectors = X[sv]
        # print("Support Vectors Shape:- ", self.support_vectors.shape)
        # print("Support Vectors:- ", self.support_vectors)
        self.alpha = alpha[sv]
        self.sv_y = y[sv]

      

        # print("Alpha Shape:- ", self.alpha.shape)
        # print("Alpha:- ", self.alpha)
        # print(self.sv_y.shape)
        # print(self.sv_y)
        
        # Compute the intercept
        
        self.b = 0
        for n in range(len(self.alpha)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alpha * self.sv_y * K[ind[n], sv])
            
        if len(self.alpha)>0:
          self.b /= len(self.alpha)

      


        # print("Bias: " , self.b)


        # Compute the weights
        if self.kernel == linear:
          self.w = np.zeros(X.shape[1])
          for n in range(len(self.alpha)):
              self.w += self.alpha[n] * self.sv_y[n] * self.support_vectors[n]
        else:
          self.w = None
        
        
        # print(self.w.shape)
        # print("Weights: ", self.w)



    
        

    
    def multi_predict(self, X)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        
        # # Read the data
        # data = pd.read_csv(test_data_path, header=None)
        # # X = data.iloc[1:, 1:-1].values
        # X = data.iloc[1:, 2:].values
        # # print(X.shape)
        
        
        # # y = data.iloc[1:, -1].astype(float).values
        # y = data.iloc[1:, 1].astype(float).values

        # # Converting 0s to -1s
        # y[y == 0] = -1

        

        

        # print(y.shape)
        # print(y)

        if self.w is not None:
            return np.dot(X, self.w) + self.b
        # else:
        #   y_predict = np.zeros(len(X))
        #   for i in range(len(X)):
        #       s = 0
        #       for a, sv_y, sv in zip(self.alpha, self.sv_y, self.support_vectors):
        #           s += a * sv_y * self.kernel(X[i],no = 2, Y = sv)
        #       y_predict[i] = s
        #   return y_predict + self.b
        
        else:
          y_predict = np.zeros(len(X))
          for i in range(len(X)):
              s = 0
              # print(len(zip(self.alpha, self.sv_y, self.support_vectors)))
              for a, sv_y, sv in zip(self.alpha, self.sv_y, self.support_vectors):
                
                # Adding kwargs
                kwargs = self.kwargs
                kwargs['no'] = 2
                kwargs['Y'] = sv


                s += a * sv_y * self.kernel(X[i], **kwargs)
                #   s += a * sv_y * rbf_kernel2(X[i], sv, 0.1 )
              y_predict[i] = s
          return y_predict + self.b
            # return np.sign(np.dot(self.alpha * self.sv_y, self.kernel(self.support_vectors, X, **self.kwargs)) + self.b)

    

    # Calculate the accuracy and F1 scores
    def multi_score(self, test_data_path:str)->float:
        y_pred = self.predict(test_data_path)
        print(y_pred)
        # y_true = pd.read_csv(test_data_path, header=None).iloc[1:, -1].astype(float).values
        y_true = pd.read_csv(test_data_path, header=None).iloc[1:, 1].astype(float).values

        # Converting 0s to -1s
        y_true[y_true == 0] = -1

        print(y_true)
        
        
        # Calculate the accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print("Accuracy: ", accuracy)

        # Calculate the F1 score
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print("F1 score: ", f1)

        return accuracy




    
# # Testing the code
# if __name__ == "__main__":
#     # Set the hyper-parameters
#     C = 1
#     kernel = polynomial
#     kwargs = {"gamma": 0.1}
#     kwargs["degree"] = 4

#     # Create the trainer
#     trainer = Trainer(kernel, C, **kwargs)

#     # Fit the model
#     trainer.fit("/content/drive/MyDrive/COL341-A2/bi_train.csv")

#     # Calculate the accuracy
#     trainer.score("/content/drive/MyDrive/COL341-A2/bi_val.csv")
from typing import List
import numpy as np
from svm_binary import Trainer
import pandas as pd

# Import the kernels
from kernel import linear, polynomial, rbf, sigmoid, laplacian






# Accuracy Functions




# Calculating the accuracy

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Calculating the confusion matrix

def confusion_matrix(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            matrix[i, j] = np.sum(np.logical_and(y_true == i+1, y_pred == j+1))
    return matrix

# Calculating the precision

def precision(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = confusion_matrix(y_true, y_pred)
    precision = np.zeros(n_classes)
    for i in range(n_classes):
        precision[i] = matrix[i, i] / np.sum(matrix[:, i])
    return precision

# Calculating the recall

def recall(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = confusion_matrix(y_true, y_pred)
    recall = np.zeros(n_classes)
    for i in range(n_classes):
        recall[i] = matrix[i, i] / np.sum(matrix[i, :])
    return recall

# Calculating the F1 score

def f1_score(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = confusion_matrix(y_true, y_pred)
    f1_score = np.zeros(n_classes)
    for i in range(n_classes):
        f1_score[i] = 2 * matrix[i, i] / (np.sum(matrix[i, :]) + np.sum(matrix[:, i]))
    return f1_score

# Calculating the macro F1 score

def macro_f1_score(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = confusion_matrix(y_true, y_pred)
    f1_score = np.zeros(n_classes)
    for i in range(n_classes):
        f1_score[i] = 2 * matrix[i, i] / (np.sum(matrix[i, :]) + np.sum(matrix[:, i]))
    return np.mean(f1_score)

# Calculating the micro F1 score

def micro_f1_score(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = confusion_matrix(y_true, y_pred)
    f1_score = np.zeros(n_classes)
    for i in range(n_classes):
        f1_score[i] = 2 * matrix[i, i] / (np.sum(matrix[i, :]) + np.sum(matrix[:, i]))
    return np.sum(f1_score) / n_classes











# One vs One multiclass SVM
class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        
        C = self.C
        n_classes = self.n_classes
        kernel = self.kernel
        kwargs = self.kwargs
        self.svms = []
        
        # for i in range(n_classes):
        #     for j in range(i+1, n_classes):
        #         self.svms.append(Trainer(kernel, C, **kwargs))
        # print(self.svms)
        
        for j in range(1, n_classes):
            for i in range(0, j):
                self.svms.append(Trainer(kernel, C, **kwargs))
        # print(self.svms)


    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        
        #Store the trained svms in self.svms
        self._init_trainers()
        
        # Read the data
        data = pd.read_csv(train_data_path, header=None)

        # for i in range(self.n_classes):
        #     for j in range(i+1, self.n_classes):
        #         # Modify the data to fit the binary classifier

        #         # Read X and y as numpy arrays
                
        #         X = data.iloc[1:, 2:].values
        #         y = data.iloc[1:, 1].astype(float).values
                
                
        #         # Converting i to 1 and j to -1
        #         y[y == i+1] = 1
        #         y[y == j+1] = -1

        #         # Fit the data
        #         self.svms[i*(self.n_classes-1)+j].multi_fit(X, y)


        for j in range(1, self.n_classes):
            for i in range(0, j):
                # Modify the data to fit the binary classifier

                # Read X and y as numpy arrays
                
                X = data.iloc[1:, 2:].values
                y = data.iloc[1:, 1].astype(float).values
                
                
                # Select the data for class i and class j
                idx = np.logical_or(y == i+1, y == j+1)
                X = X[idx]
                y = y[idx]

                # Converting i to 1 and j to -1
                y[y == i+1] = 1
                y[y == j+1] = -1

                # Fit the data
                self.svms[j*(j-1)//2+i].multi_fit(X, y)

                







        
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels

        # Read the data
        data = pd.read_csv(test_data_path, header=None)

        # Read X and y as numpy arrays
        X = data.iloc[1:, 2:].values
        y = data.iloc[1:, 1].astype(float).values

        # Predict the labels
        n_samples = X.shape[0]
        n_classes = self.n_classes
        votes = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                # votes[:, i] += self.svms[i*(n_classes-1)+j].multi_predict(X)
                # votes[:, j] += -self.svms[i*(n_classes-1)+j].multi_predict(X)
                votes[:, i] += self.svms[j*(j-1)//2+i].multi_predict(X)
                votes[:, j] += -self.svms[j*(j-1)//2+i].multi_predict(X)
        
        return np.argmax(votes, axis=1) + 1

        
    

# One vs All multiclass SVM
class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 

        # Find the number of classes


        # Initiate the trainers
        C = self.C
        n_classes = self.n_classes
        kernel = self.kernel
        kwargs = self.kwargs
        self.svms = []
        for i in range(n_classes):
            self.svms.append(Trainer(kernel, C, **kwargs))
        # print(self.svms)
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms

        self._init_trainers()

        # Read the data
        data = pd.read_csv(train_data_path, header=None)

        for i in range(self.n_classes):
            # Modify the data to fit the binary classifier

            # Read X and y as numpy arrays
            X = data.iloc[1:, 2:].values
            y = data.iloc[1:, 1].astype(float).values

            #Print the original data
            # print("Original data for class ", i+1, ":- ", y)

            # Converting i to 1 and others to -1
            y[y == float(i+1)] = 1
            y[y != 1] = -1

            # Print the modified data
            # print("Modified data for class ", i+1, ":- ", y)
            

            # Fit the data
            self.svms[i].multi_fit(X, y)

    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels

        # Read the data
        data = pd.read_csv(test_data_path, header=None)

        # Read X and y as numpy arrays
        X = data.iloc[1:, 1:].values
        # y = data.iloc[1:, 1].astype(float).values

        # Predict the labels
        n_samples = X.shape[0]
        n_classes = self.n_classes
        
        # Printing the predicted labels for each class
        
        for i in range(n_classes):
            print("Predicted labels for class ", i+1, ":- ", self.svms[i].multi_predict(X))



        votes = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            votes[:, i] += self.svms[i].multi_predict(X)

        # print("Votes:- ", votes)
        return np.argmax(votes, axis=1) + 1
    

# # Testing the code
# if __name__ == "__main__":
#     # Test the OVA
#     # kwargs = {"degree": 4, "gamma": 0.1}
#     trainer = Trainer_OVA(linear, C=1, n_classes=10, **kwargs)
#     trainer.fit("/content/drive/MyDrive/COL341-A2/multi_train.csv")
#     print("OVA Prediction:- ", trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"))

#     #Caclulate the accuracy
#     # Print the true values
#     data = pd.read_csv("/content/drive/MyDrive/COL341-A2/multi_val.csv", header=None)
#     y = data.iloc[1:, 1].astype(float).values
    
#     ova_accuracy = accuracy(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"))
    
#     # # Calculating F1 scores
#     # ova_f1 = f1_score(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"), average=None)
#     # ova_micro_f1 = f1_score(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"), average='micro')
#     # ova_macro_f1 = f1_score(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"), average='macro')

    



    
    
#     # Test the OVO
#     # kwargs = {"degree": 4, "gamma": 0.1}
#     trainer = Trainer_OVO(linear, C=1,  n_classes=10, **kwargs)
#     trainer.fit("/content/drive/MyDrive/COL341-A2/multi_train.csv")
#     print("OVO Prediction:- ", trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"))

  
#     # # Calculating F1 scores
#     # ovo_f1 = f1_score(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"), average=None)
#     # ovo_micro_f1 = f1_score(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"), average='micro')
#     # ovo_macro_f1 = f1_score(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"), average='macro')




#     # Print the true values
#     data = pd.read_csv("/content/drive/MyDrive/COL341-A2/multi_val.csv", header=None)
#     y = data.iloc[1:, 1].astype(float).values
    
#     ovo_accuracy = accuracy(y, trainer.predict("/content/drive/MyDrive/COL341-A2/multi_val.csv"))
    
    

#     print("OVA Accuracy:- ", ova_accuracy)
#     print("OVO Accuracy:- ", ovo_accuracy)

#     # # Printing the F1 scores
#     # print("OVA F1 scores:- ", ova_f1)
#     # print("OVA Micro F1 score:- ", ova_micro_f1)
#     # print("OVA Macro F1 score:- ", ova_macro_f1)
    
#     # print("OVO F1 scores:- ", ovo_f1)
#     # print("OVO Micro F1 score:- ", ovo_micro_f1)
#     # print("OVO Macro F1 score:- ", ovo_macro_f1)

    
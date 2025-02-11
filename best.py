from typing import Union
from svm_binary import Trainer as T
from svm_multiclass import Trainer_OVO as T_ovo, Trainer_OVA as T_ova

from kernel import linear, polynomial, rbf, sigmoid, laplacian


def best_classifier_two_class()->T:
    """Return the best classifier for the two-class classification problem."""
    #TODO: implement, use best performing values for C, kernel functions and all the parameters of the kernel functions
    # Set Hyper-params
    # Create trainer = T(hyper-parameters)

    trainer = T(C=0.001, kernel=linear, n_classes=10)


    return trainer

def best_classifier_multi_class()->Union[T_ovo,T_ova]:
    """Return the best classifier for the multi-class classification problem."""
    #TODO: implement, use best performing model with optimum values for C, kernel functions and all the parameters of the kernel functions.
    # Set Hyper-params
    # Set the trainer to either of T_ovo or T_ova
    # Create trainer with hyper-parameters
    
    trainer = T_ova(C=1, kernel=linear, n_classes=10)



    return trainer

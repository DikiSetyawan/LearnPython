#Data initialization 
import numpy as np

def dataPreps(Data_x, TargetValue):
    Data_x = np.array(Data_x)
    TargetValue = np.array(TargetValue) 
    
    assert Data_x.shape[0] == TargetValue.shape[0]
    assert Data_x.shape[0] != 0
    if len(Data_x.shape) < 2 : 
        m = Data_x.shape[0]
        Data_x = Data_x.reshape(m, -1)
    return Data_x, TargetValue


def lossFunctions (PredictedValue, OriginalValue) : 
    squaredDifferetiaion = np.power(OriginalValue - PredictedValue, 2)
    return np.mean(squaredDifferetiaion)

def train (Data_x : np.ndarray, 
           TargetValue : np.ndarray,
           weightMatrices : np.ndarray) : 
    #initilize W matrix 
    
    #Start Multiplying it 
    N = np.dot(Data_x, w)
    return N
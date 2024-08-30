import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y0_train, y1_train):
################################
#  Non Editable Region Ending  #
################################

    def mapped(v):
        v = np.asarray(v)
        prod_cache = np.cumprod(1 - 2 * v[::-1])[::-1]
        prod_cache = np.concatenate([np.ones(1), prod_cache])
        a = np.zeros(64)
        a[0:31] = np.cumsum(v[1:] * prod_cache[1:32])[::-1]
        a[31:63] = v[:32]
        return a

    def my_map(X):
        return np.array([mapped(v) for v in X])

    
    X_mapped = my_map(X_train)
    
    
   
    
    
    model0 = LogisticRegression()
    model0.fit(X_mapped, y0_train)
    W0 = model0.coef_.flatten()
    b0 = model0.intercept_[0]
    
    model1 = LogisticRegression()
    model1.fit(X_mapped, y1_train)
    W1 = model1.coef_.flatten()
    b1 = model1.intercept_[0]
    
    
  
    
    
    return W0, b0, W1, b1

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################

    def mapped(v):
        v = np.asarray(v)
        prod_cache = np.cumprod(1 - 2 * v[::-1])[::-1]
        prod_cache = np.concatenate([np.ones(1), prod_cache])
        a = np.zeros(64)
        a[0:31] = np.cumsum(v[1:] * prod_cache[1:32])[::-1]
        a[31:63] = v[:32]
        return a

    return np.array([mapped(v) for v in X])

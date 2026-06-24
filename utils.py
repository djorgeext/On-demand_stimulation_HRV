import numpy as np
from scipy.stats import zscore
def cxk(rr, sint):
    """
    Calculates the scaled cross-correlation between two sequences.
    
    Parameters:
    rr (array-like): First input sequence (e.g., RR intervals).
    sint (array-like): Second input sequence.
    
    Returns:
    numpy.ndarray: Array containing the correlation values for each lag.
    """
    # Convert inputs to numpy arrays to enable element-wise math
    rr = np.asarray(rr)
    sint = np.asarray(sint)
    
    # 'bandera' translated to 'max_lag' to describe its actual purpose
    max_lag = 800
    n = len(rr)
    
    # Initialize the output array 'y' with zeros
    y = np.zeros(max_lag + 1)
    
    # Python's range(x) goes from 0 to x-1, so we use max_lag + 1
    for k in range(max_lag + 1):
        
        # Python uses 0-based indexing. 
        # rr[0:n-k] takes elements from index 0 up to (but excluding) index n-k
        rr_subset = rr[:n-k]
        
        # sint[k:n] takes elements from index k up to the end
        sint_subset = sint[k:n]
        
        # Element-wise multiplication, summation, and division
        y[k] = np.sum(rr_subset * sint_subset) / (n - k)
        
    return y
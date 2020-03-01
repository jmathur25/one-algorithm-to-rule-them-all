import numpy as np

def is_internally_separable(x_f, y, c):
    """
        Checks if an array for a feature is separable on the interior of the distribution
        x_f: feature array
        y: output
        c: class that we are identifing on the interior
    """
    data_c = x_f[y==c]
    upper_quartile = np.percentile(data_c, 75)
    lower_quartile = np.percentile(data_c, 25)
    iqr = upper_quartile - lower_quartile
    b_arr = (lower_quartile-1.5*iqr <= x_f) & (x_f <= upper_quartile+1.5*iqr)
    y_internal = y[b_arr]
    x_internal = x_f[b_arr]
    
    decision_points = [0.25, 0.4, 0.5, 0.6, 0.75]
    cutoff = 0.7 # at least this much should belong to this class in interior section
    
    for dp in decision_points:
        end = int(len(y_internal) * dp)
        if (y_internal[:end]==c).mean() >= cutoff:
            return True
    return False
            
def is_externally_separable(x_f, y):
    """
        Returns (STATUS, DESCRIPTOR)
        where STATUS is a boolean indicating whether the externals are separable
        and DESCRIPTOR describes which class dominates both ends (only if that is the case, otherise this is -1)
    """
    upper_quartile = np.percentile(x_f, 75)
    lower_quartile = np.percentile(x_f, 25)
    
    left = y[x_f <= lower_quartile]
    right = y[x_f >= upper_quartile]
    
    
    cutoff = 0.7 # at least this much should belong to this class in interior section
    ml = left.mean()
    if ml >= cutoff:
        # left side dominated by 1
        mr = right.mean()
        if mr >= cutoff:
            # right side also dominated by 1
            return False, 1
        elif (1-mr) >= cutoff:
            # right side dominated by 0
            return True, -1
        return False, -1 # left side is 1 but right side indeterminate
    
    elif (1-ml) >= cutoff:
        # left side dominated by 0
        mr = right.mean()
        if mr >= cutoff:
            # right side dominated by 1
            return True, -1
        elif (1-mr) >= cutoff:
            # right side also dominated by 0
            return False, 0
        return False, -1 # left side is 1 but right side indeterminate
    
    else:
        return False, -1 # left side is indeterminate
    
    
def check_feature_distributions(x, y):
    """
        Returns a dictionary showing which features are good for which model.
        Note that in general if something works for LR, it works for RF.
        We use LR over RF because it is 1) statistically/theoretically grounded, 
        2) has few assumptions, 3) directly interpretable as probabilitic, and 4) faster
    """
    result = {
        'lr': [],
        'rf': [],
    }
    for i in range(x.shape[1]):
        s, d = is_externally_separable(x[:,i], y)
        if s:
            # is externally separable
            result['lr'].append(i)
            result['rf'].append(i)
        else:
            if d != -1:
                # if d is not -1, that means both ends are the same value and we need to like inside the distro
                c = 1 - d # if d is 0, we look for 1 inside. And vice versa
                if is_internally_separable(x[:,i], y, c):
                    result['rf'].append(i)
    return result



import numpy as np

def is_internally_separable(x_f, y):
    """
        Checks if an array for a feature is separable on the interior of the distribution
        x_f: feature array
        y: output
        c: class that we are identifing on the interior
    """
    m = y.mean() # useful for class imbalance
    for c in [0, 1]:
        data_c = x_f[y==c]
        upper_quartile = np.percentile(data_c, 75)
        lower_quartile = np.percentile(data_c, 25)
        iqr = upper_quartile - lower_quartile
        b_arr = (lower_quartile-1.5*iqr <= x_f) & (x_f <= upper_quartile+1.5*iqr)
        y_internal = y[b_arr]
        x_internal = x_f[b_arr]

        decision_points = [0.25, 0.5, 0.75]
        
        ym = m if c == 1 else 1 - m
        delta = 0.1 # 0.1 off the mean
        if ym > 0.85:
            delta = (1 - ym) / 3
        elif ym < 0.15:
            delta = ym / 3

        for dp in decision_points:
            end = int(len(y_internal) * dp)
            v = (y_internal[:end]==c).mean()
            if v >= ym + delta:
                return True
            elif v <= ym - delta:
                return True
    return False

def analyze_side(side, cutoff):
    m = side.mean()
    if m >= cutoff:
        return True, 1
    elif (1 - m) >= cutoff:
        return True, 0
    return False, -1
            
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
    bl, dl = analyze_side(left, cutoff)
    br, dr = analyze_side(right, cutoff)
    
    # currently we shall say LR is not good only if both ends are not determined or both ends dominated by same class
    if bl and br:
        # both sides were dominated
        if dl == dr:
            # same class both sides
            return False, [dl, dr]
        else:
            # diff class on both ends
            return True, [dl, dr]
    elif bl and not br:
        # left dominated
        return True, [dl, dr]
    elif not bl and br:
        # right dominated
        return True, [dl, dr]
    else:
        # neither dominated
        return False, [dl, dr]
    
def check_feature_distributions(x, y):
    """
        Returns a dictionary showing which features are good for which model.
        Note that in general if something works for LR, it works for RF.
        We use LR over RF because it is 1) statistically/theoretically grounded, 
        2) has few assumptions, 3) directly interpretable as probabilitic, and 4) faster
    """
    result = {
        'logit': [],
        'internal_sep': [],
        'half-logit': []
    }
    for i in range(x.shape[1]):
        s, d = is_externally_separable(x[:,i], y)
        if s:
            if d[0] != -1 and d[1] != -1:
                # is externally separable
                result['logit'].append(i)
            else:
                # is half separable
                result['half-logit'].append(i)
        if is_internally_separable(x[:,i], y) or is_internally_separable(x[:,i], y):
            result['internal_sep'].append(i)
    return result


def sample_over_feature_score(df):
    features = df.shape[1]
    samples = df.shape[0]
    if samples < 1000:
        penalty = 0.10
    else:
        penalty = 0
    raw_score = samples/features
    if (raw_score > 0 and raw_score < 1):
        return 0.1
    if (raw_score >= 1 and raw_score < 10):
        return 0.2
    if (raw_score >= 10 and raw_score < 100):
        return 0.5
    if (raw_score >= 100 and raw_score < 200):
        return 0.90 - penalty
    if (raw_score >= 200):
        return 0.95 - penalty
    
    return 0

class ModelChooser:
    def __init__(self):
        pass
    
    def decide(self, x, y, verbose=True):
        # feature distros: 0.8, n/f: 0.2
        rf_score = 0
        
        # metric n/f
        rf_penalty = 1 - sample_over_feature_score(x)
        rf_score -= 0.2 * rf_penalty
        
        # feature distros
        result = check_feature_distributions(x, y)
        if verbose:
            print("Feature distribution summary:\n", result, end='\n\n')
        rf_bonus = self._get_feature_distribution_score(result, x.shape[1])
        rf_score += 0.8 * rf_bonus
        
        if rf_score > x.shape[1] * 0.2:
            print("Random Forest is the better option")
        else:
            print("Logistic Regression is the better option")
        
    
    def _get_feature_distribution_score(self, result, nf):
        """
            What positively affects RF score:
             - no logit + internal_sep = non-linear
             - low number of logits/half-logits
            
        """
        score = 0
        for f in result['internal_sep']:
            if f not in result['logit']:
                score += 0.3
        
        # weights logit over half-logit, sees if it is under a third of the number of features
        if 0.66*len(result['logit']) + 0.33*len(result['half-logit']) < 0.2 * nf:
            score += 0.5
        
        return score


class FDMeta:
    """
    Build the feedback dictionary object that will be utilized during the 
    interaction
    """

    def __init__(self, fd, a, b, support, vios, vio_pairs):
        self.lhs = fd.split(' => ')[0][1:-1].split(', ')
        self.rhs = fd.split(' => ')[1].split(', ')
        self.alpha = a
        self.alpha_history = [a]
        self.beta = b
        self.beta_history = [b]
        self.conf = (a / (a+b))
        self.support = support
        self.vios = vios
        self.vio_pairs = vio_pairs


# Calculate the initial probability mean for the FD
def initialPrior(mu, variance):
    beta = (1 - mu) * ((mu * (1 - mu) / variance) - 1)
    alpha = (mu * beta) / (1 - mu)
    return alpha, beta

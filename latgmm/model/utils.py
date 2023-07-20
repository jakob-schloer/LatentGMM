"""Util functions for VAE models."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


# Functions for Gaussians
# ======================================================================================
def sample_gaussian(mu, log_v):
    """Sampling from a gaussian using the reparametrization trick.

    z = mu + v**2 * eps | eps ~ N(0,1)

    Args:
        mu (torch.Tensor): (batch, ...) mean from the encoder's latent space
        log_v (torch.Tensor): (batch, ...) log variance from the encoder's latent space
    
    Return:
        sample (tensor): (batch, ...)
    """
    std = torch.exp(0.5*log_v)  # standard deviation
    eps = torch.randn_like(std)
    sample = mu + (eps * std)  # sampling as if coming from the input space

    return sample


def compute_precision_cholesky(covariances, covariance_type='diag'):
    """Compute the Cholesky decomposition of the precisions.

    Args:
        covariances (torch.Tensor): (n_components, n_features, n_features) Covariance matrix.

    Returns
        precisions_chol (torch.Tensor): (n_components, n_features, n_features) 
            The cholesky decomposition of the precision matrix.
    """
    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = torch.empty((n_components, n_features, n_features),
                                      device=covariances.device)

        for k, covariance in enumerate(covariances):
            try:
                cov_chol = torch.linalg.cholesky(covariance, upper=False)
            except torch.linalg.LinAlgError:
                raise ValueError("Covariance matrix is ill-defined, i.e. not positive definite.")

            precisions_chol[k] = torch.linalg.solve_triangular(
                cov_chol, torch.eye(n_features, device=covariances.device), upper=False
            ).T

    elif covariance_type == 'diag':
        precisions_chol = 1. / torch.sqrt(covariances)

    else:
        raise ValueError(f"Covariance type '{covariance_type}' is not supported.")
    
    
    return precisions_chol


def compute_log_det_cholesky(matrix_chol, covariance_type='diag'):
    """Compute the log-det of the cholesky decomposition of matrices.

    Args:
        matrix_chol (torch.Tensor): (n_components, n_features, n_features)
            Cholesky decompositions of the matrices.
        covariance_type (str): {'full', 'diag'}

    Returns
        log_det_chol (torch.Tensor): (n_components,)
            The log determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, n_features, _ = matrix_chol.shape
        log_det_chol = torch.sum(torch.log(
            matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]
            ), 1)
    elif covariance_type == 'diag':
        log_det_chol = (torch.sum(torch.log(matrix_chol), dim=1))
    else:
        raise ValueError(f"Covariance type '{covariance_type}' is not supported.")
    
    return log_det_chol


def log_multivariate_gaussian(X, means, precisions_chol, covariance_type='diag'):
    """Estimate the log Gaussian probability.

    Args:
        X (torch.Tensor): (n_samples, n_features) Observations.
        means (torch.Tensor): (n_components, n_features) Means of Gaussian.
        precisions_chol (torch.Tensor): (n_components, n_features, n_features)
            Precision matrix.
        covariance_type (str): {'full', 'diag'}
        
    Returns
        log_prob (torch.Tensor): (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    # det(precision_chol) is half of det(precision)
    log_det = compute_log_det_cholesky(precisions_chol,
                                       covariance_type=covariance_type)
    # log of Gaussian exponent 
    if covariance_type == 'full':
        log_prob = torch.empty((n_samples, n_components), device=X.device)
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = torch.matmul(X, prec_chol) - torch.matmul(mu, prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (torch.sum((means ** 2 * precisions), 1) -
                    2. * torch.matmul(X, (means * precisions).T) +
                    torch.matmul(X ** 2, precisions.T))
    else:
        raise ValueError(f"Covariance type '{covariance_type}' is not supported.")

    pi = torch.tensor(np.pi, device=X.device)

    return -.5 * (n_features * torch.log(2 * pi) + log_prob) + log_det

# Functions for Gaussian mixture models
# ======================================================================================
def log_gaussian_mixture(X, means, precision_chol, pi,
                               covariance_type='diag'):
    """
    Computes log probability of a Gaussian mixture.

       log(p(c|z)) = log(p(z|c) p(c)) - log(p(z))
                   = log( pi * N(z; m, v)) - log( sum_k pi * N(z; m, v) )

    Args:
    	X (tensor): (n_samples, n_features) Observations
    	means (tensor): (n_components, n_features) Mixture means
    	precision_chol (tensor): (n_components, n_features, n_features) Mixture covariances
        pi (tensor): (n_components) Mixture coefficient

    Return:
    	log_prob: tensor: (n_samples, n_components): log probability of each sample
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    # log(p(z|c) p(c)) = log( pi * N(z; m, v))
    log_probs = (log_multivariate_gaussian(X, means, precision_chol, covariance_type) 
                 + torch.log(pi.repeat(n_samples, 1)))

    # log(p(z)) = log( sum_k pi * N(z; m, v) )
    log_p_z = log_sum_exp(log_probs, -1).unsqueeze(-1)
    
    return log_probs - log_p_z 


# Numerical stable helper functions
# ======================================================================================
def log_mean_exp(x, dim):
	"""
	Compute the log(mean(exp(x), dim)) in a numerically stable manner

	Args:
		x (tensor): Arbitrary tensor
		dim (int): Dimension along which mean is computed

	Return:
		_ (tensor): log(mean(exp(x), dim))
	"""
	return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
	"""
	Compute the log(sum(exp(x), dim)) in a numerically stable manner

	Args:
		x (tensor): Arbitrary tensor
		dim (int): Dimension along which sum is computed

	Return:
		_ (tensor): log(sum(exp(x), dim))
	"""
	max_x = torch.max(x, dim)[0]
	new_x = x - max_x.unsqueeze(dim).expand_as(x)
	return max_x + (new_x.exp().sum(dim)).log()


# Metrics
# ======================================================================================
def neg_log_likelihood(x, x_hat, dist='gaussian'):
    """
    Calculate the negative log likelihood of the p(x|z), i.e. the reconstruction loss.

    Args:
        x (tensor): (batch, x_dim) Input data
        x_hat (tensor): (batch, x_dim) Reconstruction of input data
        dist (str): Distribion of p(x|z), e.g. 'gaussian', 'bernoulli', 'laplace'

    Return:
        neg_ll (tensor): (batch)
    """
    # reconstruction loss
    if dist == 'bernoulli':
        rec_loss = F.binary_cross_entropy(x_hat, x, reduction='none')
    elif dist == 'gaussian':
        rec_loss = F.mse_loss(x_hat, x, reduction='none')
    elif dist == 'laplace':
        rec_loss = F.l1_loss(x_hat, x, reduction='none')
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    
    # (batch)
    buff = torch.flatten(rec_loss, start_dim=1)
    # TODO: Sum or Mean
    rec_loss = torch.sum(buff, dim=-1) 

    return rec_loss


def kl_normal(q_m, q_logv, p_m, p_logv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension.
 
    kl(q|p) = 0.5 * (log(p_v) - log(q_v) + (q_v**2 + (q_m - p_m)**2)/ p_v**2 - 1)

    Args:
        q_m (tensor): (batch, dim): q mean
        q_logv (tensor): (batch, dim): q log variance
        p_m (tensor): (batch, dim): p mean
        p_logv (tensor): (batch, dim): p log variance

    Return:
        kl (tensor):(batch,): kl between each sample
    """
    element_wise = 0.5 * (2*p_logv - 2*q_logv + q_logv.exp() / p_logv.exp()
                          + (q_m - p_m).pow(2) / p_logv.exp() - 1)
    kl = element_wise.sum(-1)
    return kl    


def kl_standard_normal(p_m, p_logv):
    """
    KL-divergence of standard normal prior and gaussian posterior.
    KL-Divergence = - 0.5 * sum(1 + 2*log(sigma^2) - mu^2 - sigma^2)

    Args:
        p_m: (batch, dim_z)the mean from the latent vector
        p_logv: (batch, dim_z) log variance from the latent vector
    Return:
        kl (tensor):(batch,): kl between each sample
    """
    # KL divergence of a gaussian and a standard normal gaussian
    kl = -0.5 * (1 + 2*p_logv - p_m.pow(2) - p_logv.exp()).sum(dim=-1)
    return kl


def gm_log_likelihood(x, means, precisions, weights, covariance_type='diag'):
    """Gaussian mixture log likelihood, i.e. log( sum_c p(z|c) p(c))

    Args:
        x (torch.Tensor): Sample data of dimension (n_samples, n_features) 
        means (torch.Tensor): Means of GM with dimension (n_components, n_features)
        precisions (torch.Tensor): Covariances of GM 
            with dimension (n_components, n_features, n_features)
        weights (torch.Tensor): Weights of GM with dimension (n_components) 
        covariance_type (str, optional): Covariance type. Defaults to 'diag'.

    Returns:
        log_ll (float): log likelihood of GM
    """
    log_prob = log_multivariate_gaussian(x, means, precisions,
                                         covariance_type=covariance_type)
    # Log weights
    # if logits use: nn.functional.softmax(self.logits_pi, dim=-1)
    log_weights = torch.log(
        weights
    )
    log_probs = log_prob + log_weights

    # log(p(z)) = log( sum_k pi * N(z; m, v) )
    log_p_z = log_sum_exp(log_probs, -1).unsqueeze(-1)
    log_ll = log_p_z.mean()

    return log_ll


def gm_bic(x, means, precisions, weights, covariance_type='diag'):
    """Bayesian information criterion for the Gaussian mixture in the latent space.

    Args:
        x (torch.Tensor): Sample data of dimension (n_samples, n_features) 
        means (torch.Tensor): Means of GM with dimension (n_components, n_features)
        precisions (torch.Tensor): Covariances of GM 
            with dimension (n_components, n_features, n_features)
        weights (torch.Tensor): Weights of GM with dimension (n_components) 
        covariance_type (str, optional): Covariance type. Defaults to 'diag'.

    Returns:
        bic (float): The lower the better.
    """
    # Number of GM parameters
    n_components, n_features = means.shape
    if covariance_type == 'full':
        cov_params = n_components * n_features * (n_features + 1) / 2.
    elif covariance_type == 'diag':
        cov_params = n_components * n_features
    mean_params = n_features * n_components
    n_gm_parameters = (cov_params + mean_params + n_components - 1)

    return (-2 * gm_log_likelihood(x, means, precisions, weights, covariance_type='diag') 
            * x.shape[0] 
            + n_gm_parameters * np.log(x.shape[0]))


# Pytorch NN helper functions
# ======================================================================================
def get_num_param(model):
    """Return number of trainable parameters.

    Args:
        model (nn.Module): Model

    Returns:
        [int]: Number of parameters
    """
    num_param = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)
    return num_param


def load_mlp(fname):
    """Load trained network."""
    mlp = torch.load(fname)
    print(f"Loaded mlp {fname}.")

    return mlp


def select_state_dict(state_dict, pattern='encode.',
                      replace_pattern=''):
    """Find pattern in dictionary keys and replace the pattern.

    Args:
        state_dict (dict): Input dictionary
        pattern (str, optional): Pattern to find. Defaults to 'encode.'.
        replace_pattern (str, optional): Replace pattern by. Defaults to ''.

    Returns:
        (dict) Output dictionary with replaced keys.
    """
    new_state_dict = {}
    for key, item in state_dict.items():
        if key.find(pattern) != -1:
            if replace_pattern is not None:
                new_key = key.replace(pattern, replace_pattern)
            else:
                new_key = key
            new_state_dict[new_key] = item
    return new_state_dict


# VAE helper functions
# ======================================================================================
def get_reconstructions(model, dataset, indices, batch=64):
    """Get reconstruction of inputs specified by indices.

    Args:
        model (nn.Module): [description]
        dataset (): [description]
        indices (list): Indices in dataset
        batch (int): Create batches to not run out of memory

    Returns:
        dict: Output of forward model.
    """
    # Batch encoding for smaller data
    for b in np.arange(0, len(indices), batch):
        batch_indices = indices[b : b + batch] if b + batch < len(indices) else indices[b:]
        labels = []
        for i, idx in enumerate(batch_indices):
            x, l = dataset[idx]

            if len(x.shape) != 1: # for 3 dim input
                x = x.unsqueeze_(0)

            if i == 0:
                datapoints = x.to(model.device)
            else:
                datapoints = torch.cat(
                    (datapoints, x.to(model.device)), dim=0
                )
            labels.append(l)
        x_hat, z_given_x, q_m, q_logv = model.forward(datapoints)

        # Convert torch.Tensor to numpy.ndarray
        datapoints = datapoints.cpu().detach().numpy()
        labels = np.array(labels)         
        x_hat = x_hat.cpu().detach().numpy()        
        z_given_x = z_given_x.cpu().detach().numpy()         
        try:
            q_m = q_m.cpu().detach().numpy()         
            q_logv = q_logv.cpu().detach().numpy()        
        except:
            pass
        
        if b == 0:
            rec = {'x':      datapoints,
                   'labels': labels, 
                    'x_hat': x_hat,
                    'z':     z_given_x, 
                    'q_m':   q_m, 
                    'q_logv':q_logv}
        else:
            rec['x'] = np.vstack([rec['x'], datapoints])
            rec['x_hat'] = np.vstack([rec['x_hat'], x_hat])
            rec['labels'] = np.concatenate([rec['labels'], labels])
            rec['z'] = np.vstack([rec['z'], z_given_x])
            rec['q_m'] = np.vstack([rec['q_m'], q_m])
            rec['q_logv'] = np.vstack([rec['q_logv'], q_logv])
        

    return rec


def latent_space_pca(model, train_loader, n_components=2):
    """Create PCA of latent space. 

    Args:
        model (_type_): VAE model with model.forward() function.
        train_loader (torch.Dataloader): Dataloader of training data. 

    Returns:
        pca (sklearn.decomposition.PCA): PCA of latent space. 
    """
    for i, (x, l) in enumerate(train_loader):
        x_hat, z_given_x, q_m, q_logv = model.forward(x.to(model.device))
        z = (z_given_x.cpu().detach().numpy() if i ==0 
             else np.vstack([z, z_given_x.cpu().detach().numpy()]))

    pca = PCA(n_components=n_components)
    pca.fit(z)
    print(f"Explained variance: {pca.explained_variance_ratio_}")

    return pca


def classifier(model, dataloader):
    """Classify data into learned clusters.

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_

    Returns:
        c_pred (np.ndarray): Predicted cluster assignment. 
        c_target (np.ndarray): Labeling of data. 
    """
    for i, (x, l) in enumerate(dataloader):
        p_c_given_x = model.get_p_c_given_x(x.to(model.device))
        gamma = (p_c_given_x.cpu().detach().numpy() if i ==0 
             else np.vstack([gamma, p_c_given_x.cpu().detach().numpy()]))
        label = (l.cpu().detach().numpy() if i ==0 
             else np.concatenate([label, l.cpu().detach().numpy()]))
    
    c_pred = np.argmax(gamma, axis=-1)

    return c_pred, label
    

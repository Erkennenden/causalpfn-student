"""HyperSCI numerical helpers; see SOURCE.md for small-group guards."""
import torch
import numpy as np
import torch.nn.functional as F

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]

    x = x.reshape(nx, -1)
    y = y.reshape(ny, -1)

    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, min(10.0 / (nx * ny), 0.99))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        #row = row.cuda()
        #col = col.cuda()
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        #temp_term = temp_term.cuda()
        #a = a.cuda()
        #b = b.cuda()
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            #u = u.cuda()
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        #v = v.cuda()
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        #D = D.cuda()
        D = D.to(device)

    return D, Mlam

def get_hyperedge_attr(features, hyperedge_index, type='mean'):
    # input: features: tensor N x F; hyperedge_index: 2 x |sum of all hyperedge size|
    # return hyperedge_attr: tensor, M x F
    #features = torch.FloatTensor([[0, 0.1, 0.2], [1.1, 1.2, 1.3], [2., 2.1, 2.2], [3.1,3.2,3.3], [4, 4.1, 4.2], [5,5,5]])
    #hyperedge_index = torch.LongTensor([[0,1,0,3,4,5,1],[0,0,1,1,1,2,2]])
    if type == 'mean':
        # hyperedge_attr = features[hyperedge_index[0]]  # |sum of all hyperedge size| x F
        # index_start =  # M, the start index of every hyperedge
        # hyperedge_attr = torch.tensor_split(hyperedge_attr, index_start)  #
        hyperedge_attr = None
        samples = features[hyperedge_index[0]]
        labels = hyperedge_index[1]

        labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        hyperedge_attr = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
        hyperedge_attr = hyperedge_attr / labels_count.float().unsqueeze(1)
    return hyperedge_attr

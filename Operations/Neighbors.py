import numpy as np
import scipy


def identify_nearest_neighbors_hyperbolic_q3(nntbham):
    nz_rowind, nz_colind = nntbham.nonzero()
    return nz_rowind, nz_colind


def identify_next_nearest_neighbors_hyperbolic_q3(nntbham):
    adjacency_sq = nntbham @ nntbham
    adjacency_sq = adjacency_sq - scipy.sparse.diags_array(adjacency_sq.diagonal().astype(np.float64),
                                                           offsets=0, format='csr')
    adjacency_sq.eliminate_zeros()
    nz_rowind, nz_colind = adjacency_sq.nonzero()
    return nz_rowind, nz_colind


def get_neighbors_of_site(site_index, relevant_nz_rowind, relevant_nz_colind):
    rel_mask = (relevant_nz_rowind == site_index)
    return relevant_nz_colind[rel_mask]

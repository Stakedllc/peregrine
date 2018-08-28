import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._validation import validate_graph

include 'parameters.pxi'



def bellman_ford(csgraph, directed=True, indices=None,
                 return_predecessors=False,
                 unweighted=False):
    """
    bellman_ford(csgraph, directed=True, indices=None, return_predecessors=False,
                 unweighted=False)
    Compute the shortest path lengths using the Bellman-Ford algorithm.
    The Bellman-ford algorithm can robustly deal with graphs with negative
    weights.  If a negative cycle is detected, an error is raised.  For
    graphs without negative edge weights, dijkstra's algorithm may be faster.
    .. versionadded:: 1.5.8
    Parameters
    ----------
    csgraph : array, matrix, or sparse matrix, 2 dimensions
        The N x N array of distances representing the input graph.
    directed : bool, optional
        If True (default), then find the shortest path on a directed graph:
        only move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i]
    indices : array_like or int, optional
        if specified, only compute the paths for the points at the given
        indices.
    return_predecessors : bool, optional
        If True, return the size (N, N) predecesor matrix
    unweighted : bool, optional
        If True, then find unweighted distances.  That is, rather than finding
        the path between each point such that the sum of weights is minimized,
        find the path such that the number of edges is minimized.
    Returns
    -------
    dist_matrix : ndarray
        The N x N matrix of distances between graph nodes. dist_matrix[i,j]
        gives the shortest distance from point i to point j along the graph.
    predecessors : ndarray
        Returned only if return_predecessors == True.
        The N x N matrix of predecessors, which can be used to reconstruct
        the shortest paths.  Row i of the predecessor matrix contains
        information on the shortest paths from point i: each entry
        predecessors[i, j] gives the index of the previous node in the
        path from point i to point j.  If no path exists between point
        i and j, then predecessors[i, j] = -9999
    Notes
    -----
    This routine is specially designed for graphs with negative edge weights.
    If all edge weights are positive, then Dijkstra's algorithm is a better
    choice.
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import bellman_ford
    >>> graph = [
    ... [0, 1 , 2, 0],
    ... [0, 0, 0, 1],
    ... [2, 0, 0, 3],
    ... [0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
      (0, 1)	1
      (0, 2)	2
      (1, 3)	1
      (2, 0)	2
      (2, 3)	3
    >>> dist_matrix, predecessors = bellman_ford(csgraph=graph, directed=False, indices=0, return_predecessors=True)
    >>> dist_matrix
    array([ 0.,  1.,  2.,  2.])
    >>> predecessors
    array([-9999,     0,     0,     1], dtype=int32)
    """
    #------------------------------
    # validate csgraph and convert to csr matrix
    csgraph = validate_graph(csgraph, directed, DTYPE,
                             dense_output=False)
    N = csgraph.shape[0]

    #------------------------------
    # intitialize/validate indices
    if indices is None:
        indices = np.arange(N, dtype=ITYPE)
    else:
        indices = np.array(indices, order='C', dtype=ITYPE)
        indices[indices < 0] += N
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValueError("indices out of range 0...N")
    return_shape = indices.shape + (N,)
    indices = np.atleast_1d(indices).reshape(-1)

    #------------------------------
    # initialize dist_matrix for output
    dist_matrix = np.empty((len(indices), N), dtype=DTYPE)
    dist_matrix.fill(np.inf)
    dist_matrix[np.arange(len(indices)), indices] = 0

    #------------------------------
    # initialize predecessors for output
    if return_predecessors:
        predecessor_matrix = np.empty((len(indices), N), dtype=ITYPE)
        predecessor_matrix.fill(NULL_IDX)
    else:
        predecessor_matrix = np.empty((0, N), dtype=ITYPE)

    if unweighted:
        csr_data = np.ones(csgraph.data.shape)
    else:
        csr_data = csgraph.data

    if directed:
        ret = _bellman_ford_directed(indices,
                                     csr_data, csgraph.indices,
                                     csgraph.indptr,
                                     dist_matrix, predecessor_matrix)
    else:
        raise ValueError('This implementation of bellman_ford is only valid for directed graphs')

    if ret >= 0:
        # todo: negative cycle
        pass
        # raise NegativeCycleError("Negative cycle detected on node %i" % ret)

    if return_predecessors:
        return (dist_matrix.reshape(return_shape),
                predecessor_matrix.reshape(return_shape))
    else:
        return dist_matrix.reshape(return_shape)


cdef int _bellman_ford_directed(
            np.ndarray[ITYPE_t, ndim=1, mode='c'] source_indices,
            np.ndarray[DTYPE_t, ndim=1, mode='c'] csr_weights,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csr_indices,
            np.ndarray[ITYPE_t, ndim=1, mode='c'] csr_indptr,
            np.ndarray[DTYPE_t, ndim=2, mode='c'] dist_matrix,
            np.ndarray[ITYPE_t, ndim=2, mode='c'] pred):
    cdef unsigned int Nind = dist_matrix.shape[0]
    cdef unsigned int N = dist_matrix.shape[1]
    cdef unsigned int i, j, k, j_source, count

    cdef DTYPE_t d1, d2, w12

    cdef int return_pred = (pred.size > 0)

    for i in range(Nind):
        j_source = source_indices[i]

        # relax all edges N-1 times
        for count in range(N - 1):
            # relax all edges for one iteration over N
            for j in range(N):
                d1 = dist_matrix[i, j]
                for k in range(csr_indptr[j], csr_indptr[j + 1]):
                    w12 = csr_weights[k]
                    d2 = dist_matrix[i, csr_indices[k]]
                    if d1 + w12 < d2:
                        dist_matrix[i, csr_indices[k]] = d1 + w12
                        if return_pred:
                            pred[i, csr_indices[k]] = j

        # check for negative-weight cycles
        for j in range(N):
            d1 = dist_matrix[i, j]
            for k in range(csr_indptr[j], csr_indptr[j + 1]):
                w12 = csr_weights[k]
                d2 = dist_matrix[i, csr_indices[k]]
                if d1 + w12 + DTYPE_EPS < d2:
                    return j_source

    return -1
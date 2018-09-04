from .cbellman_ford import bellman_ford_directed
import numpy as np
cimport numpy as np

from scipy import sparse
from .exchange_matrix import ExchangeMatrixBuilder
from cryptosockets import BaseExchange


include 'parameters.pxi'


cdef class ExchangeGraph:
    cdef readonly BaseExchange exchange
    cdef readonly dict currencies_by_index
    cdef readonly dict indices_by_currency
    cdef public sparse.csr_matrix csr

    def __init__(self, BaseExchange exchange, dict snapshots, ):
        """
        :param exchange: an instance of a subclass of CryptoSockets.BaseExchange
        :param snapshots: A list of un-formatted order book snapshots
        """
        self.exchange = exchange

        cdef ExchangeMatrixBuilder exch_matrix = ExchangeMatrix(exchange, snapshots)
        self.currencies_by_index = exch_matrix.currencies_by_index
        self.indices_by_currency = exch_matrix.currencies_by_index
        self.csr = exch_matrix.csr

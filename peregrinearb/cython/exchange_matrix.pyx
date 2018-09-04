import numpy as np
cimport numpy as np

from scipy import sparse
from cryptosockets import BaseExchange


cdef class ExchangeMatrixBuilder:
    cdef readonly BaseExchange exchange
    cdef readonly dict currencies_by_index
    cdef readonly dict indices_by_currency
    cdef readonly sparse.csr_matrix attr_matrix
    cdef int len_curr

    def __init__(self, BaseExchange exchange, ):
        """
        :param exchange: an instance of a subclass of CryptoSockets.BaseExchange
        """
        if not exchange.currencies_loaded:
            raise ValueError('exchange does not have currencies loaded')
        if not exchange.markets_loaded:
            raise ValueError('exchange does not have markets loaded')
        self.exchange = exchange

        self._init_matrices()

    cpdef _init_matrices(self, BaseExchange exchange, ):
        cdef dict indices_by_currency = {}
        cdef dict currencies_by_index = {}
        cdef int len_curr = len(exchange.currencies)
        cdef int max_curr_len = max(*map(lambda x: len(x), exchange.currencies))
        cdef np.dtype attr_dtype = np.dtype([
            ('base', np.dtype('U'.format(max_curr_len))),
            ('quote', np.dtype('U'.format(max_curr_len))),
            ('no_fee_rate', np.float64),
            ('depth', np.float64),
            ('volume', np.float64),
            ('side', np.dtype('U4')),
            ('fee', np.float32),
        ])
        cdef sparse.dok_matrix attr_matrix = sparse.dok_matrix((len_curr, len_curr), dtype=attr_dtype)
        # matrix that will be converted to csr matrix on call to construct_from_snapshots
        cdef sparse.dok_matrix dok_matrix = sparse.dok_matrix((len_curr, len_curr), dtype=np.float64)

        cdef int currency_count = 0
        cdef str symbol, base, quote
        cdef int base_index, quote_index
        cdef dict market_data
        for symbol, market_data in exchange.markets.items():
            base, quote = symbol.split('/')
            base_index, currency_count = \
                self._get_currency_index(base, currency_count, currencies_by_index, indices_by_currency)
            quote_index, currency_count = \
                self._get_currency_index(quote, currency_count, currencies_by_index, indices_by_currency)

            attr_matrix[(base_index, quote_index)] =

        self.currencies_by_index = indices_by_currency
        self.indices_by_currency = currencies_by_index
        self.len_curr = len_curr

    cpdef construct_from_snapshots(self, BaseExchange exchange, dict snapshots, ):
        """
        Constructs a scipy.sparse.dok_matrix m from the data in snapshots, then converts m to scipy.sparse.csr_matrix
        and returns m.

        The value at (x, y) in m is the -log of the rate at which x converts to y. For selling on market X/Y, 
        this is -log(highest bid * (1 - fee)), found at m[x, y]. For buying on X/Y, this is 
        -log(1 / lowest ask * (1 - fee)), found at m[y, x].
        """
        cdef int len_curr = len(exchange.currencies)
        cdef sparse.dok_matrix dok_matrix = sparse.dok_matrix((len_curr, len_curr), dtype=np.float64)
        cdef dict currencies_by_index = {}
        cdef dict indices_by_currency = {}
        cdef int currency_count = 0
        for i, (symbol, order_book) in enumerate(snapshots.items()):
            cdef dict ask_bid_data = exchange.format_first_order_book_elements(order_book)
            if ask_bid_data['ask'] is None:
                continue
            if ask_bid_data['bid'] is None:
                continue

            base, quote = symbol.split('/')
            base_index, currency_count = \
                self._get_currency_index(base, currency_count, currencies_by_index, indices_by_currency)
            quote_index, currency_count = \
                self._get_currency_index(quote, currency_count, currencies_by_index, indices_by_currency)

            cdef float fee = exchange.markets(symbol)['taker_fee']

            dok_matrix[(base_index, quote_index)] = ask_bid_data['bid'] * (1 - fee)
            dok_matrix[(quote_index, base_index)] = 1 / ask_bid_data['ask'] * (1 - fee)

        self.len_curr = len_curr
        self.currencies_by_index = currencies_by_index
        self.indices_by_currency = indices_by_currency
        cdef sparse.csr_matrix csr = dok_matrix.tocsr()
        csr.data = -np.log10(csr.data)
        self.csr = csr

    cpdef get_edge(self, str base, str quote):
        # todo
        cdef int base_index = self.indices_by_currency[base]
        cdef int quote_index = self.indices_by_currency[quote]


    def _get_currency_index(self, currency, currency_count, currencies_by_index: dict, indices_by_currency ):
        if currency in indices_by_currency:
            index = indices_by_currency[currency]
        else:
            index = currency_count
            indices_by_currency[currency] = currency_count
            currencies_by_index[currency_count] = currency
            currency_count += 1
        return index, currency_count

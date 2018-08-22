import numpy as np
from scipy import sparse


class SymbolNotLoadedError(Exception):
    # todo: use CryptoSockets.SymbolNotLoadedError
    pass


class ExchangeMatrix:

    def __init__(self, exchange, snapshots: dict, ):
        """
        
        :param exchange: an instance of a subclass of CryptoSockets.BaseExchange with
        :param snapshots: A list of un-formatted order book snapshots
        """
        if not exchange.currencies_loaded:
            raise ValueError('exchange does not have currencies loaded')
        if not exchange.markets_loaded:
            raise ValueError('exchange does not have markets loaded')
        self.exchange = exchange
        self.currencies_by_index = None
        self.indices_by_currency = None
        self.dok = None
        self.csr = None
        self.construct_from_snapshots(snapshots)

    def construct_from_snapshots(self, snapshots, ):
        """
        Constructs a scipy.sparse.dok_matrix m from the data in snapshots, then converts m to scipy.sparse.csr_matrix
        and returns m.

        The value at (x, y) in m is the rate at which x converts to y. For selling on market X/Y, this is the
        highest bid * (1 - fee), found at m[x, y]. For buying on X/Y, this is 1 / lowest ask * (1 - fee),
        found at m[y, x].
        """
        len_curr = len(self.exchange.currencies)
        dok_matrix = sparse.dok_matrix((len_curr, len_curr), dtype=np.float64)
        currencies_by_index = {}
        indices_by_currency = {}
        currency_count = 0
        for i, (symbol, order_book) in enumerate(snapshots.items()):
            ask_bid_data = self.exchange.format_first_order_book_elements(order_book)
            if ask_bid_data['ask'] is None:
                continue
            if ask_bid_data['bid'] is None:
                continue

            base, quote = symbol.split('/')
            base_index, currency_count = \
                self._get_currency_index(base, currency_count, currencies_by_index, indices_by_currency)
            quote_index, currency_count = \
                self._get_currency_index(quote, currency_count, currencies_by_index, indices_by_currency)

            fee = self.exchange.markets(symbol)['taker_fee']

            dok_matrix[(base_index, quote_index)] = ask_bid_data['bid'] * (1 - fee)
            dok_matrix[(quote_index, base_index)] = 1 / ask_bid_data['ask'] * (1 - fee)

        self.currencies_by_index = currencies_by_index
        self.indices_by_currency = indices_by_currency
        self.dok = dok_matrix
        self.csr = dok_matrix.tocsr()

    def _get_currency_index(self, currency, currency_count, currencies_by_index: dict, indices_by_currency: dict, ):
        if currency in indices_by_currency:
            index = indices_by_currency[currency]
        else:
            index = currency_count
            indices_by_currency[currency] = currency_count
            currencies_by_index[currency_count] = currency
            currency_count += 1
        return index, currency_count

import logging
from decimal import Decimal


__all__ = [
    'wss_add_market',
    'wss_update_graph',
]


adapter = logging.getLogger(__name__)


def wss_add_market(graph, symbol, market_data):
    base, quote = symbol.split('/')
    graph.add_edge(base, quote, weight=Decimal('Inf'), depth=Decimal('Inf'), market_name=symbol,
                   fee=market_data['taker_fee'], volume=Decimal('0'), no_fee_rate=-Decimal('Inf'), trade_type='SELL')
    graph.add_edge(quote, base, weight=Decimal('Inf'), depth=Decimal('Inf'), market_name=symbol,
                   fee=market_data['taker_fee'], volume=Decimal('0'), no_fee_rate=Decimal('Inf'), trade_type='BUY')


def wss_update_graph(graph, symbol, side, price, volume, *args):
    """Must take args because for L3 order books order id is given"""
    base, quote = symbol.split('/')
    fee_scalar = 1 - graph[base][quote]['fee']
    # if the ask was updated
    if side == 'sell':
        opp_could_exist = price < graph[quote][base]['no_fee_rate']
        graph[quote][base]['weight'] = -(fee_scalar * 1 / price).ln()
        graph[quote][base]['depth'] = -(volume * price).ln()
        graph[quote][base]['volume'] = volume
        graph[quote][base]['no_fee_rate'] = price
    # if the bid was updated
    else:
        opp_could_exist = price > graph[base][quote]['no_fee_rate']
        graph[base][quote]['weight'] = -(price * fee_scalar).ln()
        graph[base][quote]['depth'] = -volume.ln()
        graph[base][quote]['volume'] = volume
        graph[base][quote]['no_fee_rate'] = price

    return opp_could_exist

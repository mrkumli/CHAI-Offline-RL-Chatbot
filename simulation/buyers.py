"""Transparent, heuristic buyer agents. Not the original paper's evaluators."""
from dataclasses import dataclass


@dataclass
class Buyer:
    style: str
    reservation: float  # max buyer will pay as fraction of list price

    def first_offer(self):
        ratios = {'cooperative': .85, 'stingy': .62, 'sensitive': .75}
        return min(self.reservation, ratios[self.style])

    def reply(self, seller_price, turn_number):
        """Buyer can accept, reject (end), or counteroffer (continue)."""
        # More generous or tolerant buyers accept offers further below their max.
        tolerance = {'cooperative': .98, 'stingy': .88, 'sensitive': .94}[self.style]
        if seller_price <= self.reservation * tolerance:
            return 'accept', seller_price
        if turn_number >= 5 or seller_price > self.reservation * 1.28:
            return 'reject', None
        # Raise the buyer's previous bid gradually, never above reservation.
        improvement = {'cooperative': .06, 'stingy': .015, 'sensitive': .035}[self.style]
        return 'offer', min(self.reservation, self.first_offer() + improvement * turn_number)


RESERVATIONS = {'cooperative': 1.02, 'stingy': .74, 'sensitive': .88}


def make_buyer(style, reservation=None):
    if style not in RESERVATIONS:
        raise ValueError(f'Unknown buyer: {style}')
    return Buyer(style, RESERVATIONS[style] if reservation is None else reservation)

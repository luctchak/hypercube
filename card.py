from dataclasses import dataclass

@dataclass
class Card:
    symbol: str
    quantity: int
    color: tuple[int, int, int]
    fill: str
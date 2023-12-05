import random
from card import Card
from enums import PlayerOrder, Faction
from unit import Unit
from structure import Structure
from point import Point
from board import Board
from typing import List

class Player:
    def __init__(self, faction: Faction, deck: List[Card], order: PlayerOrder):
        self.board: Board = None
        self.faction = faction
        self.order = order
        self.max_mana = 3 if order == PlayerOrder.FIRST else 4
        self.current_mana = self.max_mana
        self.strength = 20
        self.front_line = 4 if order == PlayerOrder.FIRST else 0
        self.replacable = True

        self.deck = deck
        random.shuffle(self.deck)

        for i, card in enumerate(self.deck):
            card.player = self
            card.weight = 1 if i == 0 else self.deck[i - 1].weight * 1.6 + 100

        self.hand: List[Card] = []
        self.fill_hand()

        self.actions = []

    def __eq__(self, other):
        return self.order == other.order

    @property
    def opponent(self):
        return self.board.remote

    def draw(self, amount=1):
        for _ in range(amount):
            choice = random.choices(self.deck, weights=[card.weight for card in self.deck])[0]
            choice.weight = 1
            self.hand.append(choice)
            self.deck.remove(choice)

    def fill_hand(self):
        self.draw(4 - len(self.hand))

    def reweight(self):
        for card in self.deck:
            card.weight = card.weight * 1.6 + 100

    def discard(self, target: Card):
        self.reweight()
        self.hand.remove(target)

        if not target.is_single_use:
            self.deck.append(target)

    def play(self, index: int, position: Point | None):
        target = self.hand[index]
        self.board.add_to_history(target)
        self.discard(target)

        if isinstance(target, Unit) or isinstance(target, Structure):
            self.board.set(position, target.copy())
            self.board.at(position).play(position)
        else:
            target.play(position)

    def cycle(self, target: Card):
        self.discard(target)
        self.draw()

    def deal_damage(self, amount: int):
        self.strength -= amount

    def heal(self, amount: int):
        self.strength += amount
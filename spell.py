from enums import Faction
from card import Card
from target import Target
from point import Point
from colorama import Back, Style

class Spell(Card):
    def __init__(self, faction: Faction, cost: int, required_targets: Target | None = None):
        super().__init__()
        self.faction = faction
        self.cost = cost
        self.required_targets = required_targets

    def __repr__(self):
        color = Back.BLUE if self.player == self.player.board.local else Back.RED

        return f"{color}{self.card_id}          {Style.RESET_ALL}"

    def play(self, position: Point | None = None):
        if self.required_targets is None or position in self.player.board.get_targets(self.required_targets):
            self.activate_ability(position)
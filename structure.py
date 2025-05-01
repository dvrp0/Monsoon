from card import Card
from enums import Faction, TriggerType
from typing import List
from point import Point
from colorama import Back, Fore, Style

class Structure(Card):
    def __init__(self, faction: Faction, cost: int, strength: int, triggers: List[TriggerType]=[TriggerType.TURN_START]):
        super().__init__()
        self.faction = faction
        self.cost = cost
        self.strength = strength
        self.triggers = triggers
        self.position: Point = None
        self.damage_taken = 0
        self.damage_source = None

    def __eq__(self, other):
        return isinstance(other, Structure) and self.card_id == other.card_id and self.player == other.player and self.position == other.position

    def __repr__(self):
        fore = Fore.BLUE if self.player == self.player.board.local else Fore.RED
        color = Back.BLUE if self.player == self.player.board.local else Back.RED
        strength = f"♥{self.strength}{' ' if self.strength < 10 else ''}"

        return f"{fore}{self.position}{Style.RESET_ALL} {color}{self.card_id} {strength}      {Style.RESET_ALL}"
    
    @property
    def is_at_turn_start(self):
        return TriggerType.TURN_START in self.triggers

    @property
    def is_at_turn_end(self):
        return TriggerType.TURN_END in self.triggers
    
    @property
    def is_on_play(self):
        return TriggerType.ON_PLAY in self.triggers

    def play(self, position: Point):
        self.position = position
        self.player.board.set(self.position, self)

        if self.is_on_play:
            self.activate_ability(source=self)

    def deal_damage(self, amount: int, pending_destroy=False, source: Card | None = None):
        if self.strength - amount < 0:
            amount = self.strength

        self.damage_taken = amount
        self.damage_source = source
        self.strength -= amount

        if not pending_destroy and self.strength <= 0:
            self.destroy(source)

        return amount

    def destroy(self, source: Card | None = None):
        self.damage_taken = self.strength
        self.damage_source = source
        self.player.board.set(self.position, None)
        self.player.board.calculate_front_line(self.player.board.current_player.opponent)

    def heal(self, amount: int):
        self.strength += amount

    def respawn(self, position: Point, strength: int):
        if type(self) is Structure:
            structure = Structure(
                self.faction,
                self.cost,
                strength,
                self.triggers
            )
        else:
            structure = self.__class__()
            structure.strength = strength

        structure.player = self.player
        structure.position = position

        self.player.board.set(position, structure)
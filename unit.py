from card import Card
from enums import Faction, UnitType, Phase, TriggerType, StatusEffect
from typing import List
from point import Point
from colorama import Back, Fore, Style

class Unit(Card):
    def __init__(self, faction: Faction, unit_types: List[UnitType], cost: int, strength: int, movement: int, trigger: TriggerType = None, fixedly_forward=False):
        super().__init__()
        self.faction = faction
        self.unit_types = unit_types
        self.cost = cost
        self.strength = strength
        self.movement = movement
        self.fixedly_forward = fixedly_forward
        self.trigger: TriggerType = trigger
        self.status_effects = []
        self.position: Point = None

    def __eq__(self, other):
        return isinstance(other, Unit) and self.card_id == other.card_id and self.player == other.player and self.position == other.position

    def __repr__(self):
        fore = Fore.BLUE if self.player == self.player.board.local else Fore.RED
        color = Back.BLUE if self.player == self.player.board.local else Back.RED
        strength = f"♥{min(99, self.strength)}{' ' if self.strength < 10 else ''}"
        is_frozen = f"{Back.LIGHTCYAN_EX} {color}" if self.is_frozen else ' '
        is_poisoned = f"{Back.GREEN} {color}" if self.is_poisoned else ' '
        is_vitalized = f"{Back.LIGHTGREEN_EX} {color}" if self.is_vitalized else ' '
        is_confused = f"{Back.YELLOW} {color}" if self.is_confused else ' '
        is_disabled = f"{Back.MAGENTA} {color}" if self.is_disabled else ' '

        return f"{fore}{self.position}{Style.RESET_ALL} " \
            f"{color}{self.card_id} {strength} {is_frozen}{is_poisoned}{is_vitalized}{is_confused}{is_disabled}{Style.RESET_ALL}"

    @property
    def is_frozen(self):
        return StatusEffect.FROZEN in self.status_effects

    @property
    def is_poisoned(self):
        return StatusEffect.POISONED in self.status_effects

    @property
    def is_vitalized(self):
        return StatusEffect.VITALIZED in self.status_effects

    @property
    def is_confused(self):
        return StatusEffect.CONFUSED in self.status_effects

    @property
    def is_disabled(self):
        return StatusEffect.DISABLED in self.status_effects

    def play(self, position: Point):
        self.position = position
        self.player.board.set(self.position, self)
        self.set_path()

        if self.trigger == TriggerType.ON_PLAY:
            self.activate_ability()

        self.move()

    def set_path(self):
        destinations: List[Point] = []
        position = self.position
        status_effects_cached = list(self.status_effects)
        is_turn_start = self.player.board.phase == Phase.TURN_START

        for _ in range(1 if is_turn_start else self.movement):
            destination = Point(position.x, position.y + (-1 if self.player == self.player.board.local else 1))
            next = self.player.board.at(destination)

            if StatusEffect.CONFUSED in status_effects_cached:
                if position.x == 0:
                    choices = [1]
                elif position.x == 3:
                    choices = [-1]
                else:
                    choices = [-1, 1]

                destination = Point(position.x + int(self.player.random.choice(choices)), position.y)
                status_effects_cached.remove(StatusEffect.CONFUSED)
            elif not is_turn_start and not self.fixedly_forward and destination.y > -1 and (next is None or next.player == self.player):
                # 상대 기지 앞이 아닌 위치에서 앞에 아무것도 없거나 앞이 아군으로 막혀 있다면 옆을 살펴보기
                left_position = Point(position.x - 1, position.y)
                left = self.player.board.at(left_position) if position.x > 0 else None
                right_position = Point(position.x + 1, position.y)
                right = self.player.board.at(right_position) if position.x < 3 else None

                # inward -> outward
                if position.x <= 1:
                    if right is not None and right.player != self.player and right_position not in destinations:
                        destination = right_position
                    elif left is not None and left.player != self.player and left_position not in destinations:
                        destination = left_position
                elif position.x >= 2:
                    if left is not None and left.player != self.player and left_position not in destinations:
                        destination = left_position
                    elif right is not None and right.player != self.player and right_position not in destinations:
                        destination = right_position

            destinations.append(destination)
            position = destination

        self.path = destinations

    def move(self):
        if self.player.board.phase == Phase.TURN_START:
            if self.is_poisoned:
                self.deal_damage(1)
            elif self.is_vitalized:
                self.heal(1)

            if self.is_frozen:
                self.unfreeze()

                return

        if self.trigger == TriggerType.BEFORE_MOVING and not self.is_disabled and len(self.path) > 0:
            self.activate_ability()

        if self.is_frozen: # If frozen during ability, skip the rest
            return

        for destination in self.path:
            if destination.y < 0 or destination.y > 4: # To base
                if self.trigger == TriggerType.BEFORE_ATTACKING and not self.is_disabled:
                    self.activate_ability(destination)

                self.player.opponent.deal_damage(self.strength)

                if self.player.opponent.strength > 0:
                    self.destroy()

                return

            target = self.player.board.at(destination)
            is_attacked = False

            if target is not None and target.player == self.player and destination.y < self.position.y:
                return

            if target is not None and (self.is_confused or target.player != self.player):
                if self.trigger == TriggerType.BEFORE_ATTACKING and not self.is_disabled:
                    self.activate_ability(destination)

                target = self.player.board.at(destination) # target may have changed
                if target is not None:
                    target_strength_cached = target.strength
                    target_on_death_pending = isinstance(target, Unit) and target.trigger == TriggerType.ON_DEATH and not target.is_disabled
                    local_on_death_pending = self.trigger == TriggerType.ON_DEATH and not self.is_disabled

                    target.deal_damage(self.strength, target_on_death_pending)
                    self.deal_damage(target_strength_cached, local_on_death_pending)

                    if target.strength <= 0 and target_on_death_pending:
                        target.destroy() # defender triggers first
                    if self.strength <= 0 and local_on_death_pending:
                        self.destroy()

                    is_attacked = True

            if self.player.board.at(destination) is None and self.strength > 0:
                self.player.board.set(self.position, None)
                self.position = destination
                self.player.board.set(destination, self)

                if self.player.front_line > destination.y:
                    self.player.front_line = max(1, destination.y)

                if is_attacked and self.trigger == TriggerType.AFTER_ATTACKING and not self.is_disabled:
                    self.activate_ability()

                if self.is_confused:
                    self.deconfuse()

    def deal_damage(self, amount: int, pending_destroy=False):
        self.strength -= amount

        if not pending_destroy and self.strength <= 0:
            self.destroy()
        elif self.trigger == TriggerType.AFTER_SURVIVING and self.strength > 0:
            self.player.board.push_trigger(self.activate_ability)
            self.player.board.pop_trigger()

    def destroy(self):
        self.player.board.set(self.position, None)
        self.path = []

        if self.trigger == TriggerType.ON_DEATH:
            self.player.board.push_trigger(self.activate_ability)
            self.player.board.pop_trigger()

        self.player.board.calculate_front_line(self.player.board.remote)

    def reduce(self, amount: int):
        self.strength = max(1, self.strength - amount)

    def heal(self, amount: int):
        self.strength += amount

    def freeze(self):
        self.status_effects.append(StatusEffect.FROZEN)

    def unfreeze(self):
        self.status_effects.remove(StatusEffect.FROZEN)

    def poison(self):
        if self.is_vitalized:
            self.unvitalize()

        self.status_effects.append(StatusEffect.POISONED)

    def unpoison(self):
        self.status_effects.remove(StatusEffect.POISONED)

    def vitalize(self):
        if self.is_poisoned:
            self.unpoison()

        self.status_effects.append(StatusEffect.VITALIZED)

    def unvitalize(self):
        self.status_effects.remove(StatusEffect.VITALIZED)

    def confuse(self):
        self.status_effects.append(StatusEffect.CONFUSED)

    def deconfuse(self):
        self.status_effects.remove(StatusEffect.CONFUSED)

    def disable(self):
        if type(self).activate_ability != Card.activate_ability:
            self.status_effects.append(StatusEffect.DISABLED)

    def enable(self):
        if self.is_disabled:
            self.status_effects.remove(StatusEffect.DISABLED)

    def gain_speed(self, amount: int):
        self.movement += amount
        self.set_path() # gaining speed is only possible on play
        self.movement -= amount

    def command(self):
        fixedly_forward_cache = self.fixedly_forward
        self.fixedly_forward = True

        self.set_path()
        self.move()

        self.fixedly_forward = fixedly_forward_cache

    def convert(self):
        self.player = self.player.opponent
        self.set_path()

    def pull(self, position: Point): # pull this unit to position
        points = []

        if position.y < self.position.y: # from front
            points = [Point(self.position.x, y) for y in range(self.position.y - 1, -1, -1)]
        elif position.y > self.position.y: # from behind
            points = [Point(self.position.x, y) for y in range(self.position.y + 1, 5)]
        elif position.x < self.position.x: # from left
            points = [Point(x, self.position.y) for x in range(self.position.x - 1, -1, -1)]
        elif position.x > self.position.x: # from right
            points = [Point(x, self.position.y) for x in range(self.position.x + 1, 4)]

        for point in points:
            if self.player.board.at(point) is not None:
                return

            self.player.board.set(self.position, None)
            self.position = point
            self.player.board.set(point, self)

        if self.player.front_line > self.position.y:
            self.player.front_line = max(1, self.position.y)

    def push(self, position: Point): # push this unit from position
        points = []

        if position.y < self.position.y: # from front
            points = [Point(self.position.x, y) for y in range(self.position.y + 1, 5)]
        elif position.y > self.position.y: # from behind
            points = [Point(self.position.x, y) for y in range(self.position.y - 1, -1, -1)]
        elif position.x < self.position.x: # from left
            points = [Point(x, self.position.y) for x in range(self.position.x + 1, 4)]
        elif position.x > self.position.x: # from right
            points = [Point(x, self.position.y) for x in range(self.position.x - 1, -1, -1)]

        for point in points:
            if self.player.board.at(point) is not None:
                return

            self.player.board.set(self.position, None)
            self.position = point
            self.player.board.set(point, self)

        if self.player.front_line > self.position.y:
            self.player.front_line = max(1, self.position.y)

    def teleport(self, destination: Point):
        if self.player.board.at(destination) is None:
            self.player.board.set(self.position, None)
            self.position = destination
            self.player.board.set(destination, self)

            if self.player.front_line > destination.y:
                self.player.front_line = max(1, destination.y)

            self.set_path()
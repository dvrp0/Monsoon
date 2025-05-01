import numpy as np
from enums import Faction, Phase, UnitType
from card import Card
from collections.abc import Callable
from colorama import Back, Fore, Style
from unit import Unit
from structure import Structure
from target import Context, Target
from typing import TYPE_CHECKING, List, Tuple
from point import Point

if TYPE_CHECKING:
    from player import Player

class Board:
    def __init__(self, local: "Player", remote: "Player", random: np.random.RandomState):
        self.board: List[List[Unit | Structure | None]] = [[None for _ in range(4)] for _ in range(5)]
        self.local = local
        self.remote = remote
        self.current_player = local
        self.history: List[Card] = []
        self.random = random
        self.triggers: List[Tuple[Callable, Card | None]] = []
        self.is_resolving_trigger = False
        self.phase = Phase.PLAY

        self.local.board = self
        self.remote.board = self

    def __repr__(self):
        rows = []

        for y in range(5):
            row = "  ".join(str(self.board[y][x]) if self.board[y][x] is not None else
                f"{Fore.LIGHTBLACK_EX}({x}, {y}) {' ' * 14}{Style.RESET_ALL}" for x in range(4))
            local_front = f"{Fore.BLUE}▶{Style.RESET_ALL}" if self.local.front_line == y else " "
            remote_front = f"{Fore.RED}◁{Style.RESET_ALL}" if self.remote.front_line == y else " "
            rows.append(f"{local_front} {row} {remote_front}")

        remote = [f"{' ' * 47}{Back.RED} {self.remote.order}: {self.remote.strength} {Style.RESET_ALL}"]
        local = [f"{' ' * 47}{Back.BLUE} {self.local.order}: {self.local.strength} {Style.RESET_ALL}"]
        history = [", ".join(f"[{'local' if card.player == self.local else 'remote'}: {card.card_id}]" for card in self.history[-4:])]

        return "\n".join(remote + rows + local + history)

    def push_trigger(self, trigger: Callable, source: Card | None = None):
        self.triggers.append((trigger, source))

    def pop_trigger(self):
        if len(self.triggers) == 0 or self.is_resolving_trigger:
            return

        trigger, source = self.triggers.pop()

        if trigger is not None:
            trigger(source=source)

    def at(self, position: Point):
        if position.is_base:
            return self.local if position.y == 5 else self.remote

        if not position.is_valid:
            return None

        return self.board[position.y][position.x]

    def set(self, position: Point, entity: Unit | Structure | None):
        self.board[position.y][position.x] = entity

        if entity is not None:
            entity.position = position

    def clear(self):
        self.board = [[None for _ in range(4)] for _ in range(5)]
        self.local.front_line = 4
        self.remote.front_line = 0

    def calculate_front_line(self, player: "Player"):
        if player == self.local:
            for y in range(5):
                if any(self.board[y][x] is not None and self.board[y][x].player == player for x in range(4)):
                    self.local.front_line = max(1, y) # farthest front line is 1
                    break

                self.local.front_line = 4
        elif player == self.remote:
            for y in range(4, -1, -1):
                if any(self.board[y][x] is not None and self.board[y][x].player == player for x in range(4)):
                    self.remote.front_line = min(3, y) # farthest front line is 3
                    break

                self.remote.front_line = 0

    def flip(self):
        temp = self.local
        self.local = self.remote
        self.remote = temp

        self.local.front_line = 4 - self.local.front_line
        self.remote.front_line = 4 - self.remote.front_line

        self.board = [row[::-1] for row in self.board[::-1]]
        for y in range(5):
            for x in range(4):
                if self.board[y][x] is not None:
                    self.board[y][x].position = Point(x, y)

    def to_next_turn(self):
        self.phase = Phase.TURN_END
        self.current_player.fill_hand()

        for structure in [self.at(tile) for tile in self.get_targets(None, Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))]:
            if structure.is_at_turn_end:
                structure.activate_ability(structure.position, structure)

        self.calculate_front_line(self.local)
        self.calculate_front_line(self.remote)

        self.current_player.max_mana += 1
        self.local.current_mana = self.local.max_mana
        self.remote.current_mana = self.remote.max_mana

        self.phase = Phase.TURN_START
        self.current_player = self.current_player.opponent
        self.current_player.replacable = True
        self.current_player.leftmost_movable = True

        for structure in [self.at(tile) for tile in self.get_targets(None, Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))]:
            if structure.is_at_turn_start:
                structure.activate_ability(structure.position, structure)

        for unit in [self.at(tile) for tile in self.get_targets(None, Target(Target.Kind.UNIT, Target.Side.FRIENDLY))]:
            unit.set_path()
            unit.move()

        self.phase = Phase.PLAY

    def get_targets(self, context: Context | None, target: Target) -> List[Point]:
        if context is None:
            context = Context()

        if context.pov is None:
            context.pov = self.current_player

        tiles = []

        # Trigger order is the movement order of non-attacker player
        y_range = range(5) if context.pov == self.local else range(4, -1, -1)
        x_range = range(4) if context.pov == self.local else range(3, -1, -1)

        for y in y_range:
            for x in x_range:
                entity = self.board[y][x]

                if entity is None or entity.strength <= 0:
                    continue

                is_unit = isinstance(entity, Unit)
                is_structure = isinstance(entity, Structure)

                if is_unit:
                    type_matches = True if target.unit_types is None else any(type in entity.unit_types for type in target.unit_types)
                    exclude_type_matches = True if target.exclude_unit_types is None else not any(type in entity.unit_types for type in target.exclude_unit_types)
                    non_hero_matches = True if not target.non_hero else UnitType.HERO not in entity.unit_types
                    status_matches = True if target.status_effects is None else any(status in entity.status_effects for status in target.status_effects)
                    exclude_status_matches = True if target.exclude_status_effects is None else not any(status in entity.status_effects for status in target.exclude_status_effects)

                strength_matches = True if target.strength_limit is None else entity.strength <= target.strength_limit
                unit_matches = is_unit and type_matches and exclude_type_matches and non_hero_matches and status_matches and exclude_status_matches and strength_matches
                structure_matches = is_structure and strength_matches

                kind_matches = ((target.kind == Target.Kind.ANY and (unit_matches or structure_matches)) or
                    (target.kind == Target.Kind.UNIT and unit_matches) or
                    (target.kind == Target.Kind.STRUCTURE and structure_matches))
                side_matches = (target.side == Target.Side.ANY or
                    (target.side == Target.Side.FRIENDLY and entity.player == context.pov) or
                    (target.side == Target.Side.ENEMY and entity.player != context.pov))

                if kind_matches and side_matches:
                    tiles.append(Point(x, y))

        if target.include_base:
            friendly = Point(-1, 5) if context.pov == self.local else Point(-1, -1)
            enemy = Point(-1, -1) if context.pov == self.local else Point(-1, 5)

            if target.side in [Target.Side.FRIENDLY, Target.Side.ANY]:
                tiles.append(friendly)
            
            if target.side in [Target.Side.ENEMY, Target.Side.ANY]:
                tiles.append(enemy)

        if context.exclude is not None and context.exclude in tiles:
            tiles.remove(context.exclude)

        return tiles

    def get_front_tiles(self, context: Context, target: Target = None) -> List[Point]:
        if context.pov is None:
            context.pov = self.current_player

        y_range = range(context.position.y - 1, -1, -1) if context.pov == self.local else range(context.position.y + 1, 5)
        tiles = [Point(context.position.x, i) for i in y_range]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles or (point.is_base and target.include_base)]

        tiles.sort(key=lambda t: t.y, reverse=context.pov == self.local)

        return tiles

    def get_behind_tiles(self, context: Context, target: Target = None) -> List[Point]:
        if context.pov is None:
            context.pov = self.current_player

        y_range = range(context.position.y + 1, 5) if context.pov == self.local else range(context.position.y - 1, -1, -1)
        tiles = [Point(context.position.x, i) for i in y_range]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles or (point.is_base and target.include_base)]

        tiles.sort(key=lambda t: t.y, reverse=context.pov == self.remote)

        return tiles

    def get_side_tiles(self, context: Context, target: Target = None) -> List[Point]:
        tiles = [
            Point(context.position.x - 1, context.position.y),
            Point(context.position.x + 1, context.position.y)
        ]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles]

        return [tile for tile in tiles if tile.is_valid]

    def get_row_tiles(self, context: Context, target: Target = None) -> List[Point]:
        tiles = [Point(i, context.position.y) for i in range(0, 4)]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles]

        return [tile for tile in tiles if tile.is_valid]

    def get_column_tiles(self, context: Context, target: Target = None) -> List[Point]:
        tiles = [Point(context.position.x, i) for i in range(0, 5)]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles or (point.is_base and target.include_base)]

        return [tile for tile in tiles if tile.is_valid or (tile.is_base and target.include_base)]

    def get_bordering_tiles(self, context: Context, target: Target = None) -> List[Point]:
        tiles = [
            Point(context.position.x - 1, context.position.y),
            Point(context.position.x + 1, context.position.y),
            Point(context.position.x, context.position.y - 1),
            Point(context.position.x, context.position.y + 1)
        ]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles or (point.is_base and target.include_base)]

        return [tile for tile in tiles if tile.is_valid or (tile.is_base and target.include_base)]

    def get_surrounding_tiles(self, context: Context, target: Target = None) -> List[Point]:
        tiles = [
            Point(context.position.x - 1, context.position.y),
            Point(context.position.x - 1, context.position.y - 1),
            Point(context.position.x - 1, context.position.y + 1),
            Point(context.position.x + 1, context.position.y),
            Point(context.position.x + 1, context.position.y - 1),
            Point(context.position.x + 1, context.position.y + 1),
            Point(context.position.x, context.position.y - 1),
            Point(context.position.x, context.position.y + 1)
        ]

        if target is not None:
            targets = self.get_targets(context, target)
            tiles = [point for point in targets if point in tiles or (point.is_base and target.include_base)]

        return [tile for tile in tiles if tile.is_valid or (tile.is_base and target.include_base)]

    def spawn_token_unit(self, player: "Player", position: Point, strength: int, types: List[UnitType] = None):
        types = types or [UnitType(self.random.choice(list(UnitType)))]

        token = Unit(Faction.NEUTRAL, types, 0, strength, 1)
        token.player = player
        token.position = position

        types_id = "".join([str(type.value) for type in types])
        token.card_id = f"f{types_id.zfill(3)}"

        self.set(position, token)
        self.calculate_front_line(player)

        return self.at(position)

    def spawn_token_structure(self, player: "Player", position: Point, strength: int):
        token = Structure(Faction.NEUTRAL, 0, strength)
        token.player = player
        token.position = position
        token.card_id = "b001"

        self.set(position, token)
        self.calculate_front_line(player)

        return self.at(position)

    def add_to_history(self, card: Card):
        self.history.append(card)
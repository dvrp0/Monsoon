import numpy as np, json
from enums import *
from point import Point
from unit import Unit
from structure import Structure
from spell import Spell
from player import Player
from board import Board
from target import Target
from cards import *

class Action:
    def __init__(self, type, target=None, position: Point | None = None):
        self.type = type
        self.target = target
        self.position = position

class State:
    def __init__(self, seed=None):
        self.turn = 1
        self.players = (Player(Faction.WINTER, [], PlayerOrder.FIRST), Player(Faction.WINTER, [], PlayerOrder.SECOND))
        self.current_player = self.players[0]
        self.board = [[None for _ in range(4)] for _ in range(5)]
        self.__available_actions = None
        self.__action_mask = None
        self.winner = None

    @property
    def available_actions(self):
        if self.__available_actions is not None:
            return self.__available_actions

        place, use = [], []
        replace = [Action(ActionType.REPLACE, card) for card in self.current_player.hand]
        to_leftmost = [Action(ActionType.TO_LEFTMOST, card) for card in self.current_player.hand[1:]]

        for card in [x for x in self.current_player.hand if x.cost <= self.current_player.mana]:
            if isinstance(card, Unit) or isinstance(card, Structure):
                for y in range(self.current_player.front_line, 4):
                    for x in range(4):
                        if self.board[y][x] is None:
                            place.append(Action(ActionType.PLACE, card, Point(x, y)))
            elif isinstance(card, Spell):
                if card.required_targets is None:
                    use.append(Action(ActionType.USE, card))
                else:
                    use += [Action(ActionType.Use, card, point) for point in self.get_targetable_tiles(card.required_targets)]

        self.__available_actions = place + use + replace + to_leftmost + [Action(ActionType.PASS)]

        return self.__available_actions

    @property
    def action_mask(self):
        if self.__action_mask is not None:
            return self.__action_mask

        mask = [True] * 156

        for i in range(4):
            not_in_hand = len(self.current_player.hand) - 1 < i # i번째 카드가 없을 때
            if not_in_hand or self.current_player.hand[i].cost > self.current_player.mana:
                mask[16 * i:(16 * i) + 16] = [False] * 16 # Place
                mask[21 * i + 64:(21 * i) + 85] = [False] * 21 # Use

                if not_in_hand:
                    mask[148 + i] = False # Replace
                    if i > 0:
                        mask[152 + i] = False # To leftmost
            else:
                pass

        for y in range(self.current_player.front_line, 4):
            for x in range(4):
                if self.board[y][x] is not None:
                    mask[x + y * 4] = True

local = Player(Faction.IRONCLAD, [B304(), UA07(), U007(), U061(), U053(), U106(), U302(), U305(), U306(), U320(), UD31(), UE04()], PlayerOrder.FIRST)
remote = Player(Faction.SHADOWFEN, [S012(), UA07(), U007(), U211(), U061(), U206(), U053(), U001(), U216(), S013(), U071(), UA04()], PlayerOrder.SECOND)
board = Board(local, remote)

with open("cards.json", "r", encoding="utf-8") as f:
    cards = json.load(f)

while True:
    while True:
        print(board)
        print(f"Current player: {board.local.order}")
        print(f"Max mana: {board.local.max_mana}, Current mana: {board.local.current_mana}")
        print(f"Hand:")
        for i, card in enumerate(board.local.hand):
            for entry in cards:
                if entry["id"] == card.card_id:
                    print(f"{i}: {entry['name']} {card.card_id} {card.strength if isinstance(card, Unit) or isinstance(card, Structure) else ''} "
                        f"{card.movement if isinstance(card, Unit) else ''}")

        action = input("> ")

        if action == "end":
            print("Turn ended")
            break
        elif action.startswith("replace"):
            target = int(action.split()[1])
            board.local.cycle(board.local.hand[target])
            print(f"Replaced card {target}")
        else:
            inputs = [int(x) for x in action.split()]

            if board.local.hand[inputs[0]].cost > board.local.current_mana:
                print("Not enough mana")
            elif not isinstance(board.local.hand[inputs[0]], Spell) and inputs[2] < board.local.front_line:
                print("Can only place behind front line")
            else:
                board.local.current_mana -= board.local.hand[inputs[0]].cost
                board.local.play(inputs[0], Point(inputs[1], inputs[2]) if len(inputs) > 1 else None)

    board.to_next_turn()
from enums import *
from point import Point
from unit import Unit
from structure import Structure
from spell import Spell
from player import Player
from board import Board
from cards import *
from target import Target
from typing import Callable, List
import numpy as np

def main():
    random = np.random.RandomState(0)
    local = Player(Faction.IRONCLAD, [B304(), UA07(), U007(), U061(), U053(), U106(), U302(), U305(), U306(), U320(), UD31(), UE04()], PlayerOrder.FIRST, random)
    remote = Player(Faction.SWARM, [S012(), UA07(), U007(), U211(), U061(), U206(), U053(), U001(), U216(), S013(), U071(), UA04()], PlayerOrder.SECOND, random)
    board = Board(local, remote, random)
    player = 1

    print(board)
    

main()
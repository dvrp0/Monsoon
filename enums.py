from enum import IntEnum

class PlayerOrder(IntEnum):
    FIRST = 0
    SECOND = 1

    def opposite(self):
        return PlayerOrder((self + 1) % 2)

'''
All possible actions:

0: Place unit or structure card at index 0 of hand at (0, 4)
1: Place unit or structure card at index 0 of hand at (1, 4)
2: Place unit or structure card at index 0 of hand at (2, 4)
3: Place unit or structure card at index 0 of hand at (3, 4)
4: Place unit or structure card at index 0 of hand at (0, 3)
...
15: Place unit or structure card at index 0 of hand at (3, 1)
...
63: Place unit or structure card at index 3 of hand at (3, 1)
64: Use spell card at index 0 of hand with no target
65: Use spell card at index 0 of hand at (0, 4)
...
84: Use spell card at index 0 of hand at (3, 0)
...
147: Use spell card at index 3 of hand at (3, 0)
148: Replace card at index 0 of hand
149: Replace card at index 1 of hand
150: Replace card at index 2 of hand
151: Replace card at index 3 of hand
152: Move card at index 1 of hand to leftmost
153: Move card at index 2 of hand to leftmost
154: Move card at index 3 of hand to leftmost
155: Pass the turn
'''
class ActionType(IntEnum):
    PLACE = 0
    USE = 1
    REPLACE = 2
    TO_LEFTMOST = 3
    PASS = 4

class Faction(IntEnum):
    NEUTRAL = 0
    WINTER = 1
    SWARM = 2
    IRONCLAD = 3
    SHADOWFEN = 4

class UnitType(IntEnum):
    CONSUTRUCT = 0
    FLAKE = 1
    KNIGHT = 2
    PIRATE = 3
    RAVEN = 4
    RODENT = 5
    SATYR = 6
    TOAD = 7
    UNDEAD = 8
    VIKING = 9
    HERO = 10
    DRAGON = 11
    ELDER = 12
    FELINE = 13
    ANCIENT = 14
    PRIMAL = 15

class TriggerType(IntEnum):
    # ON_DEATH, BEFORE_ATTACKING, AFTER_ATTACKING, AFTER_SURVIVING, BEFORE_MOVING abilities
    # MUST set pov when calling board.get_targets() or its variants
    ON_PLAY = 0
    ON_DEATH = 1
    BEFORE_ATTACKING = 2
    AFTER_ATTACKING = 3
    AFTER_SURVIVING = 4
    BEFORE_MOVING = 5
    TURN_START = 6
    TURN_END = 7

class StatusEffect(IntEnum):
    FROZEN = 0
    POISONED = 1
    CONFUSED = 2
    DISABLED = 3
    VITALIZED = 4

class Phase(IntEnum):
    TURN_START = 0
    PLAY = 1
    TURN_END = 2

# Complete list of actions:
# index = 0
# for card in range(4):
#     for y in range(4, 0, -1):
#         for x in range(4):
#             print(f"{index}: Play unit or structure at index {card} of hand at ({x}, {y})")
#             index += 1
# for card in range(4):
#     print(f"{index}: Play spell card at index {card} of hand with no target")
#     index += 1
#     for y in range(4, -1, -1):
#         for x in range(4):
#             print(f"{index}: Play spell card at index {card} of hand at ({x}, {y})")
#             index += 1
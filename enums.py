from enum import IntEnum

class PlayerOrder(IntEnum):
    FIRST = 0
    SECOND = 1

    def opposite(self):
        return PlayerOrder((self + 1) % 2)

'''
All possible actions:

0: Place unit or structure card at index 0 of hand at (0, 0)
1: Place unit or structure card at index 0 of hand at (0, 1)
2: Place unit or structure card at index 0 of hand at (0, 2)
3: Place unit or structure card at index 0 of hand at (0, 3)
4: Place unit or structure card at index 0 of hand at (1, 0)
...
15: Place unit or structure card at index 0 of hand at (3, 3)
...
63: Place unit or structure card at index 3 of hand at (3, 3)
64: Use spell card at index 0 of hand with no target
65: Use spell card at index 0 of hand at (0, 0)
...
84: Use spell card at index 0 of hand at (4, 3)
...
147: Use spell card at index 3 of hand at (4, 3)
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

class TriggerType(IntEnum):
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
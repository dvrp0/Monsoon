import utils
from enums import Faction

deck = utils.generate_random_deck(Faction.WINTER)

for card in deck:
    print(type(card))
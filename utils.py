from enums import Faction
from colorama import Back, Fore, Style

def get_faction_color(faction: Faction):
    match faction:
        case Faction.NEUTRAL:
            back = Back.LIGHTBLACK_EX
            fore = Fore.WHITE
        case Faction.WINTER:
            back = Back.BLUE
            fore = Fore.WHITE
        case Faction.SWARM:
            back = Back.YELLOW
            fore = Fore.BLACK
        case Faction.IRONCLAD:
            back = Back.RED
            fore = Fore.WHITE
        case Faction.SHADOWFEN:
            back = Back.GREEN
            fore = Fore.BLACK

    return back, fore
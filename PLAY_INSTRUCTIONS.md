# Playing Against Your Trained Agent

## Quick Start

To play against your trained agent using the checkpoint file `results/evolutionary/interrupted_checkpoint.pkl`, simply run:

```bash
python play_vs_agent.py
```

Or specify a different checkpoint:

```bash
python play_vs_agent.py --checkpoint path/to/your/checkpoint.pkl
```

## Game Controls

### Basic Actions
- **Play a card**: Type `card_index x y` (e.g., `0 1 3` to play card 0 at position x=1, y=3)
- **Cast a spell**: Type `card_index x y` for targeted spells, or just `card_index` for non-targeted spells
- **Replace a card**: Type `replace card_index` (e.g., `replace 2`)
- **End turn**: Type `end`
- **Quit game**: Type `quit` or `exit`

### Understanding the Game

#### Factions
- **You (Human)**: IRONCLAD faction (Player 1, goes first)
- **Agent**: SWARM faction (Player 2, goes second)

#### Board Layout
- The board is 4x5 (4 columns, 5 rows)
- Coordinates are (x, y) where x=0-3 and y=0-4
- You can only place units/structures behind your front line
- Your base health is shown on the bottom

#### Hand Information
When it's your turn, you'll see:
- Your current/max mana
- Your hand with card indices (0, 1, 2, 3)
- Card names, IDs, costs, strength, and movement (if applicable)

### Example Turn
```
Current player: FIRST
Max mana: 5, Current mana: 4
Hand:
0: Green Prototypes U007 2 1 1
1: Gifted Recruits U061 2 1 2
2: Westwind Sailors U106 4 3 2
3: Confinement S012 2

> 0 2 4    # Play Green Prototypes at position (2, 4)
```

## Advanced Options

### Let Agent Go First
```bash
python play_vs_agent.py --agent-starts
```

### Use Different Checkpoint
```bash
python play_vs_agent.py --checkpoint results/evolutionary2/best_weights.pkl
```

## Tips for Playing

1. **Mana Management**: You start with 3 mana (4 as second player) and gain 1 each turn
2. **Unit Placement**: Units can only be placed behind your front line (the blue arrow ▶)
3. **Card Replacement**: You can replace one card per turn if you haven't used that ability yet
4. **Spell Targeting**: Some spells require targets, others don't
5. **Win Condition**: Reduce opponent's base to 0 health

## Troubleshooting

### If the script doesn't work:
1. Make sure you have all dependencies: `pip install -r requirements.txt`
2. Check that your checkpoint file exists: `ls -la results/evolutionary/interrupted_checkpoint.pkl`
3. Ensure you're in the project root directory when running the script

### Common Issues:
- **"Checkpoint file not found"**: Check the path to your checkpoint file
- **Import errors**: Make sure you're running from the project root directory
- **Game crashes**: The agent might encounter an invalid state - this is a known issue that can happen during gameplay

Enjoy playing against your trained AI! 🎮 
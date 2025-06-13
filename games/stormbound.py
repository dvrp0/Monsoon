import datetime, pathlib, numpy as np, torch, json
from .abstract_game import AbstractGame
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

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1 # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (27, 5, 4)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(156))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 8  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 400  # Maximum number of moves if game is not finished before
        self.num_simulations = 100  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = 10  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 8  # Number of channels in reward head
        self.reduced_channels_value = 8  # Number of channels in value head
        self.reduced_channels_policy = 8  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = True  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 200  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Stormbound(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        print(f"Current player: {self.env.board.local.order}")
        print(f"Max mana: {self.env.board.local.max_mana}, Current mana: {self.env.board.local.current_mana}")
        print(f"Hand:")
        for i, card in enumerate(self.env.board.local.hand):
            for entry in self.env.cards:
                if entry["id"] == card.card_id:
                    print(f"{i}: {entry['name']} {card.card_id} {card.cost} {card.strength if isinstance(card, Unit) or isinstance(card, Structure) else ''} "
                        f"{card.movement if isinstance(card, Unit) else ''}")

        while True:
            action = input("> ")
            action_representation = Action(ActionType.PASS)

            if action == "end":
                print("Turn ended")
                break
            elif action.startswith("replace"):
                if not self.env.board.local.replacable:
                    print("Already replaced")
                    continue

                target = int(action.split()[1])
                self.env.board.local.cycle(self.env.board.local.hand[target])
                action_representation = Action(ActionType.REPLACE, target)
                print(f"Replaced card {target}")
                break
            else:
                inputs = [int(x) for x in action.split()]
                card = self.env.board.local.hand[inputs[0]]

                if card.cost > self.env.board.local.current_mana:
                    print("Not enough mana")
                elif not isinstance(card, Spell) and inputs[2] < self.env.board.local.front_line:
                    print("Can only place behind front line")
                else:
                    point = Point(inputs[1], inputs[2]) if len(inputs) > 1 else None

                    action_representation = Action(ActionType.USE if isinstance(card, Spell) else ActionType.PLACE, inputs[0], point)
                    break

        return action_representation.to_int()

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return self.env.actions[action_number]

class Action:
    def __init__(self, type: ActionType, card_index: int | None = None, position: Point | None = None):
        self.type = type
        self.card_index = card_index
        self.position = position

    def to_int(self):
        result = 155

        match self.type:
            case ActionType.PLACE:
                index = 0

                for y in range(4, 0, -1):
                    for x in range(4):
                        if x == self.position.x and y == self.position.y:
                            result = 16 * self.card_index + index
                            break
                        else:
                            index += 1
            case ActionType.USE:
                if self.position is None:
                    result = 64 + 21 * self.card_index
                else:
                    index = 0

                    for y in range(4, -1, -1):
                        for x in range(4):
                            if x == self.position.x and y == self.position.y:
                                result = 65 + self.card_index * 21 + index
                                break
                            else:
                                index += 1
            case ActionType.REPLACE:
                result = 148 + self.card_index
            case ActionType.TO_LEFTMOST:
                result = 151 + self.card_index

        return result

class Stormbound:
    def __init__(self, seed):
        self.random = np.random.RandomState(seed)
        local = Player(Faction.IRONCLAD, [
            UA07(), U007(), U306(), U061(), B304(), U305(),
            U320(), U302(), U313(), UA02(), UT32(), U316()
        ], PlayerOrder.FIRST, self.random)
        remote = Player(Faction.SWARM, [
            UA07(), U007(), U001(), U053(), UE01(), U211(),
            U206(), U071(), U020(), S013(), B001(), U061()
        ], PlayerOrder.SECOND, self.random)
        self.board = Board(local, remote, self.random)
        self.player = 1

        with open("actions.txt", "r") as f:
            self.actions = f.read().splitlines()

        with open("cards.json", "r", encoding="utf-8") as f:
            self.cards = json.load(f)

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        return self.get_observation()

    def step(self, action: int):
        local_strength = self.board.local.strength
        remote_strength = self.board.remote.strength

        if action < 64: # Place
            card_index = action // 16
            index = action % 16
            excuted = False

            for y in range(4, 0, -1):
                for x in range(4):
                    if index == 0:
                        self.board.local.current_mana -= self.board.local.hand[card_index].cost
                        self.board.local.play(card_index, Point(x, y))
                        excuted = True
                        break
                    else:
                        index -= 1

                if excuted:
                    break
        elif action < 148: # Use
            card_index = (action - 64) // 21
            index = (action - 64) % 21
            excuted = False

            for y in range(4, -1, -1):
                for x in range(4):
                    if index == 0:
                        self.board.local.current_mana -= self.board.local.hand[card_index].cost
                        self.board.local.play(card_index, Point(x, y) if self.board.local.hand[card_index].required_targets is not None else None)
                        excuted = True
                        break
                    else:
                        index -= 1

                if excuted:
                    break
        elif action < 152: # Replace
            card_index = action - 148
            self.board.local.cycle(self.board.local.hand[card_index])
            self.board.local.replacable = False
        elif action < 155: # To leftmost
            card_index = action - 151
            self.board.local.hand[card_index], self.board.local.hand[0] = self.board.local.hand[0], self.board.local.hand[card_index]
            self.board.local.leftmost_movable = False

        done = self.have_winner() or len(self.legal_actions()) == 0
        reward = 1 if self.board.remote.strength <= 0 else 0

        if action == 155: # Pass
            self.player *= -1
            self.board.flip()
            self.board.to_next_turn()

        return self.get_observation(), reward, done

    '''
    Observation shape: (27, 5, 4)
    - 0: Card IDs of P1's units on the board
    - 1: Strengths of P1's units on the board
    - 2: Movements of P1's units on the board
    - 3: Bit encoded status effects(vitality, poison, confusion, freeze and disable) of P1's units on the board
    - 4: Card IDs of P1's structures on the board
    - 5: Strengths of P1's structures on the board
    - 6: P1's hand ([card ID, cost, strength, movement] * 4 + hand representing cue [32767, 32767, 32767, 32767])
    - 7-12: P1's deck ([card ID, cost, strength, movement] * n + deck representing cue [32768, 32768, 32768, 32768])
    - 13: P1's current mana (constant-valued)
    - 14: P1's base strength (constant-valued)
    - 15: P1's faction (constant-valued)
    - 16: Card IDs of P2's units on the board
    - 17: Strengths of P2's units on the board
    - 18: Movements of P2's units on the board
    - 19: Bit encoded status effects(vitality, poison, confusion, freeze and disable) of P2's units on the board
    - 20: Card IDs of P2's structures on the board
    - 21: Strengths of P2's structures on the board
    - 22: P2's current mana (constant-valued)
    - 23: P2's base strength (constant-valued)
    - 24: P2's faction (constant-valued)
    - 25: Current player (constant-valued)
    - 26: Card play history ([player, card ID, -1, -1] * 4 + history representing cue [32769, 32769, 32769, 32769])
    '''
    def get_observation(self):
        def foreach(operation: Callable[[Unit | Structure | None], Unit | Structure | None]):
            return [[operation(tile) for tile in row] for row in self.board.board]

        # P1's units
        local_unit_ids = foreach(lambda tile: int(tile) if isinstance(tile, Unit) and tile.player == self.board.local else -1)
        local_unit_strengths = foreach(lambda tile: tile.strength if isinstance(tile, Unit) and tile.player == self.board.local else -1)
        local_unit_movements = foreach(lambda tile: tile.movement if isinstance(tile, Unit) and tile.player == self.board.local else -1)
        # Bit pack status effects: vitality(1), poison(2), confusion(4), freeze(8), disable(16)
        local_unit_statuses = foreach(lambda tile: (
            (1 if isinstance(tile, Unit) and tile.player == self.board.local and tile.is_vitalized else 0) |
            (2 if isinstance(tile, Unit) and tile.player == self.board.local and tile.is_poisoned else 0) |
            (4 if isinstance(tile, Unit) and tile.player == self.board.local and tile.is_confused else 0) |
            (8 if isinstance(tile, Unit) and tile.player == self.board.local and tile.is_frozen else 0) |
            (16 if isinstance(tile, Unit) and tile.player == self.board.local and tile.is_disabled else 0)
        ) if isinstance(tile, Unit) and tile.player == self.board.local else -1)

        # P1's structures
        local_structure_ids = foreach(lambda tile: int(tile) if isinstance(tile, Structure) and tile.player == self.board.local else -1)
        local_structure_strengths = foreach(lambda tile: tile.strength if isinstance(tile, Structure) and tile.player == self.board.local else -1)

        # P1's hand
        local_hand = []
        for card in (self.board.local.hand + [None] * 4)[:4]:
            if card is None:
                local_hand.append([-1, -1, -1, -1])
            else:
                local_hand.append([
                    int(card),
                    card.cost,
                    -1 if isinstance(card, Spell) else card.strength,
                    card.movement if isinstance(card, Unit) else -1
                ])
        local_hand.append([32767] * 4)

        # P1's deck
        sorted_deck = sorted(self.board.local.deck, key=lambda x: (x.cost, x.card_id))
        local_deck = []
        for i in range(6):
            deck_layer = []
            start_idx = i * 4
            for card in (sorted_deck[start_idx:start_idx + 4] + [None] * 4)[:4]:
                if card is None:
                    deck_layer.append([-1, -1, -1, -1])
                else:
                    deck_layer.append([
                        int(card),
                        card.cost,
                        -1 if isinstance(card, Spell) else card.strength,
                        card.movement if isinstance(card, Unit) else -1
                    ])
            deck_layer.append([32768] * 4)
            local_deck.append(deck_layer)

        # P1's constant values
        local_mana = [[self.board.local.current_mana] * 4] * 5
        local_base = [[self.board.local.strength] * 4] * 5
        local_faction = [[self.board.local.faction.value] * 4] * 5

        # P2's units
        remote_unit_ids = foreach(lambda tile: int(tile) if isinstance(tile, Unit) and tile.player == self.board.remote else -1)
        remote_unit_strengths = foreach(lambda tile: tile.strength if isinstance(tile, Unit) and tile.player == self.board.remote else -1)
        remote_unit_movements = foreach(lambda tile: tile.movement if isinstance(tile, Unit) and tile.player == self.board.remote else -1)
        # Bit pack status effects: vitality(1), poison(2), confusion(4), freeze(8), disable(16)
        remote_unit_statuses = foreach(lambda tile: (
            (1 if isinstance(tile, Unit) and tile.player == self.board.remote and tile.is_vitalized else 0) |
            (2 if isinstance(tile, Unit) and tile.player == self.board.remote and tile.is_poisoned else 0) |
            (4 if isinstance(tile, Unit) and tile.player == self.board.remote and tile.is_confused else 0) |
            (8 if isinstance(tile, Unit) and tile.player == self.board.remote and tile.is_frozen else 0) |
            (16 if isinstance(tile, Unit) and tile.player == self.board.remote and tile.is_disabled else 0)
        ) if isinstance(tile, Unit) and tile.player == self.board.remote else -1)

        # P2's structures
        remote_structure_ids = foreach(lambda tile: int(tile) if isinstance(tile, Structure) and tile.player == self.board.remote else -1)
        remote_structure_strengths = foreach(lambda tile: tile.strength if isinstance(tile, Structure) and tile.player == self.board.remote else -1)

        # P2's constant values
        remote_mana = [[self.board.remote.current_mana] * 4] * 5
        remote_base = [[self.board.remote.strength] * 4] * 5
        remote_faction = [[self.board.remote.faction.value] * 4] * 5

        # Current player
        current_player = [[self.player * 99999] * 4] * 5

        # History
        history = []
        for card in ([None] * 4 + self.board.history)[-4:]:
            if card is None:
                history.append([-1, -1, -1, -1])
            else:
                history.append([
                    -99999 if card.player.order.value else 99999,
                    int(card),
                    -1,
                    -1
                ])
        history.append([32769] * 4)

        observation = np.array([
            local_unit_ids,
            local_unit_strengths,
            local_unit_movements,
            local_unit_statuses,
            local_structure_ids,
            local_structure_strengths,
            local_hand,
            *local_deck,
            local_mana,
            local_base,
            local_faction,
            remote_unit_ids,
            remote_unit_strengths,
            remote_unit_movements,
            remote_unit_statuses,
            remote_structure_ids,
            remote_structure_strengths,
            remote_mana,
            remote_base,
            remote_faction,
            current_player,
            history
        ], dtype="int32")

        # Verify shape is correct
        assert observation.shape == (27, 5, 4), f"Expected shape (27, 5, 4), got {observation.shape}"

        return observation

    def legal_actions(self):
        hand_length = len(self.board.local.hand)

        place, use = [], []
        replace = [Action(ActionType.REPLACE, card) for card in range(hand_length)] if self.board.local.replacable else []
        # to_leftmost = [Action(ActionType.TO_LEFTMOST, card) for card in range(1, hand_length)] if self.board.local.leftmost_movable else []
        to_leftmost = []

        for card in range(hand_length):
            instance = self.board.local.hand[card]

            if instance.cost > self.board.local.current_mana:
                continue

            if isinstance(instance, Unit) or isinstance(instance, Structure):
                for y in range(4, self.board.local.front_line - 1, -1):
                    for x in range(4):
                        if self.board.at(Point(x, y)) is None:
                            place.append(Action(ActionType.PLACE, card, Point(x, y)))
            elif isinstance(instance, Spell):
                if instance.required_targets is None:
                    use.append(Action(ActionType.USE, card))
                else:
                    use += [Action(ActionType.USE, card, point) for point in self.board.get_targets(None, instance.required_targets)]

        actions: List[Action] = place + use + replace + to_leftmost
        if len(actions) == len(replace):
            actions.append(Action(ActionType.PASS))

        return sorted([action.to_int() for action in actions])
        # return sorted([action.to_int() for action in actions]) if len(actions) > 0 else [155]

    def have_winner(self):
        return self.board.local.strength < 0 or self.board.remote.strength < 0

    def expert_action(self):
        '''
        1. 교체가 legal actions에 있다면 다음을 먼저 고려
        1-1. 손에서 가장 비싼 카드의 마나가 현재 마나보다 비싸다면 교체
        1-2. 손의 모든 카드의 마나가 현재 마나보다 적지만 사용할 수 없는 카드들이 있다면(배치 불가, 대상 없음 등등) 해당 카드 중 무작위로 교체
        2. 마나가 부족해질 때까지 카드 사용
        2-1. 손에서 가장 저렴한 카드부터 사용
        2-2. 만약 현재 마나와 일치하는 카드들이 있다면 그 중 무작위로 사용
        2-3. 카드를 사용하는 위치는 [전선의 맨 앞 행 중 무작위 타일, 무작위 적 유닛 또는 건물의 앞/왼쪽/오른쪽 타일 중 무작위] 중 무작위
        2-4. 사용하려는 카드가 유닛이고 적 유닛이 내 기지와 인접해 있다면 다음 기준에 따라 사용
        2-4-1. 만약 해당 유닛의 양 옆이 비어 있다면 둘 중 무작위 타일에 사용
        2-4-2. 양 옆이 막혀 있다면 원래대로 전선 뒤 무작위 타일에 사용
        3. 턴 종료
        '''

        actions = self.legal_actions()

        if any(x >= 148 and x <= 151 for x in actions):
            costs = [x.cost for x in self.board.local.hand]
            max_cost = max(costs)

            if max_cost > self.board.local.current_mana:
                index = self.random.choice([i for i, x in enumerate(costs) if x == max_cost])

                return Action(ActionType.REPLACE, index).to_int()

        playable: List[int] = []

        for i in range(4):
            if any((x >= 16 * i and x <= 16 * i + 15) or (x >= 21 * i + 64 and x <= 21 * i + 84) for x in actions):
                playable.append(i)

        if len(playable) > 0:
            costs = [self.board.local.hand[x].cost for x in playable]

            if any(x == self.board.local.current_mana for x in costs):
                index = playable[self.random.choice([i for i, x in enumerate(costs) if x == self.board.local.current_mana])]
            else:
                min_cost = min(costs)
                index = playable[self.random.choice([i for i, x in enumerate(costs) if x == min_cost])]

            target = self.board.local.hand[index]
            base_bordering_enemies = [x for x in self.board.get_targets(None, Target(Target.Kind.UNIT, Target.Side.ENEMY)) if x.y == 4]

            if isinstance(target, Spell):
                position = None if target.required_targets is None else self.random.choice(self.board.get_targets(None, target.required_targets))

                return Action(ActionType.USE, index, position).to_int()
            elif isinstance(target, Unit) and len(base_bordering_enemies) > 0:
                candidates = []

                for enemy in base_bordering_enemies:
                    if enemy.x > 0 and self.board.at(Point(enemy.x - 1, enemy.y)) is None:
                        candidates.append(Point(enemy.x - 1, enemy.y))
                    elif enemy.x < 3 and self.board.at(Point(enemy.x + 1, enemy.y)) is None:
                        candidates.append(Point(enemy.x + 1, enemy.y))

                if len(candidates) > 0:
                    return Action(ActionType.PLACE, index, self.random.choice(candidates)).to_int()
            else:
                candidates = [Point(x, self.board.local.front_line) for x in range(4) if self.board.at(Point(x, self.board.local.front_line)) is None]
                enemies = self.board.get_targets(None, Target(Target.Kind.UNIT, Target.Side.ENEMY))

                for enemy in enemies:
                    if enemy.x > 0 and enemy.y >= self.board.local.front_line and self.board.at(Point(enemy.x - 1, enemy.y)) is None:
                        candidates.append(Point(enemy.x - 1, enemy.y))
                    elif enemy.x < 3 and enemy.y >= self.board.local.front_line and self.board.at(Point(enemy.x + 1, enemy.y)) is None:
                        candidates.append(Point(enemy.x + 1, enemy.y))
                    elif enemy.y < 4 and enemy.y + 1 >= self.board.local.front_line and self.board.at(Point(enemy.x, enemy.y + 1)) is None:
                        candidates.append(Point(enemy.x, enemy.y + 1))

                if len(candidates) > 0:
                    return Action(ActionType.PLACE, index, self.random.choice(candidates)).to_int()

        return Action(ActionType.PASS).to_int()

    def render(self):
        print(self.board)
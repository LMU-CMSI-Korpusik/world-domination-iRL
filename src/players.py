"""
Different types of Players that can play in Risk. RiskPlayer implemented
while referencing pac-cap final project for Dr. Forney's CMSI 4320

Author: Kieran Ahn, Jason Douglas
Date: 11/27/2023, 12/07/2023
"""
from validators import *
from riskGame import Card, Territory, Action
from riskLogic import Board, Player
from riskNet import RiskNet
import numpy as np
import random
import torch
from constants import *
import pickle
import time
from os.path import exists
from collections import namedtuple, deque, Counter
from copy import deepcopy

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'selected_options', 'is_over'))
LastState = namedtuple('LastState', ('board', 'state'))


class RandomPlayer(Player):
    """
    A very simple Player that just makes random choices in all situations.
    """

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        return rng.choice(list(free_territories))

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        armies_placed = None
        if armies_to_place == 1:
            armies_placed = 1
        else:
            armies_placed = int(rng.integers(
                1, armies_to_place, endpoint=True))
        return (rng.choice(list(self.territories)), armies_placed)

    def attack(self, board: Board) -> tuple[Territory, Territory, int]:
        stop_attack = rng.random() * (self.occupied_territories() + 1) < 1
        occupied_territories = set(list(self.territories))

        if stop_attack:
            return (None, None, None)

        valid_attack_targets = self.get_valid_attack_targets(
            board, occupied_territories)

        if len(valid_attack_targets) == 0:
            return (None, None, None)

        target = rng.choice(list(valid_attack_targets))

        valid_bases = self.get_valid_bases(
            board, target, occupied_territories)

        base = rng.choice(list(valid_bases))

        attacking_armies = rng.integers(
            1, min(board.armies[base] - 1, 3), endpoint=True)

        return target, base, int(attacking_armies)

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        if board.armies[base] - 1 == attacking_armies:
            return attacking_armies
        return int(rng.integers(attacking_armies, board.armies[base]))

    def defend(self, board_state: Board, target: Territory, attacking_armies: int) -> int:
        if board_state.armies[target] == 1:
            return 1
        else:
            return int(rng.integers(1, 2, endpoint=True))

    def fortify(self, board: Board) -> tuple[Territory, Territory, int]:
        no_fortify = rng.random() * (self.occupied_territories() + 1) < 1

        if no_fortify:
            return (None, None, None)

        occupied_territories = set(list(self.territories))
        possible_destinations = [territory for territory, neighbors in board.territories.items()
                                 if territory in self.territories
                                 and len([neighbor for neighbor in neighbors if neighbor in self.territories and board.armies[neighbor] > 1]) != 0]

        if len(possible_destinations) == 0:
            return (None, None, None)

        destination = rng.choice(possible_destinations)
        valid_sources = self.get_valid_bases(
            board, destination, occupied_territories)

        source = rng.choice(list(valid_sources))

        fortifying_armies = None

        if board.armies[source] == 2:
            fortifying_armies = 1
        else:
            fortifying_armies = int(rng.integers(1, board.armies[source]))

        return destination, source, fortifying_armies

    def use_cards(self, board: Board, matches: list[Card]) -> tuple[Card, Card, Card]:
        if len(matches) == 0:
            return None
        return rng.choice(matches)

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        return rng.choice(potential_territories)

class HumanPlayer(Player):
    """
    A Player that takes input from the user to make decisions.
    """
    def attack(self, board: Board) -> tuple[Territory, Territory, int]:
        print('Attack')
        print('======')
        print('Territories:')
        for territory in self.territories:
            print(f'\t{territory}')
        print('')

        print('Targets:')
        for territory, neighbors in board.territories.items():
            if territory not in self.territories:
                continue
            for neighbor in neighbors:
                if neighbor not in self.territories:
                    print(f'\t{territory} -> {neighbor}')
        print('')

        target = input('Target: ')
        base = input('Base: ')
        armies = input('Armies: ')

        return (target, base, armies)
    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        print('Capture')
        print('=======')
        print('Target:')
        print(f'\t{target}')
        print('Base:')
        print(f'\t{base}')
        print('Armies:')
        print(f'\t{attacking_armies}')
        print('')
        return input('Armies: ')
        
    def defend(self, board: Board, target: Territory, attacking_armies: int) -> int:
        print('Defend')
        print('======')
        print('Target:')
        print(f'\t{target}')
        print('Armies:')
        print(f'\t{attacking_armies}')
        print('')
        print('1 or 2?')
        return input('Armies: ')
        
    def fortify():
        pass
    def use_cards():
        pass



class RiskPlayer(Player):
    """
    A player that implements RiskNet for decisionmaking.
    """

    @staticmethod
    def get_territories_mask(board: Board, valid_territories: list[Territory] | set[Territory]) -> list[int]:
        mask = [0 for _ in range(len(board.territories))]
        for territory in valid_territories:
            mask[board.territory_to_index[territory]] = 1
        return mask

    @staticmethod
    def get_territory_from_index(board: Board, territory_index: int) -> Territory:
        for territory, index in board.territory_to_index.items():
            if territory_index == index:
                return territory

    def get_reward(self, last_board: Board, last_action, current_state, board: Board) -> int:
        if board.is_win(self):
            return 1

        if any([board.is_win(player) for player in board.players if player is not self]):
            return -1

        equivalent_player = [
            player for player in last_board.players if player.name == self.name][0]

        reward = -0.1

        reward += 0.05 * max(len(self.territories) // 3, 3) / 3

        reward += sum([1 for continent in board.continents if continent.territories.issubset(
            self.territories.keys())]) * 0.01

        return reward

    def optimize(self, action: Action, board: Board):
        # This function is an exercise in sunk-cost fallacy.
        sample = random.sample(self.action_memory[action], BATCH_SIZE)
        batch = Transition(*zip(*sample))

        def is_over(state):
            territories = state[:42 * 6]
            for territory in range(6):
                if 42 == torch.sum(territories[territory:42 + territory]).item():
                    return True
            return False

        def get_action_int(to_convert):
            if to_convert is None:
                return 0 if action == Action.CARDS else len(board.territories)
            if type(to_convert) is tuple:
                return board.territory_to_index[to_convert[0]]
            if type(to_convert) is Territory:
                return board.territory_to_index[to_convert]
            if type(to_convert) is int:
                return to_convert
            if type(to_convert) is float:
                return to_convert

        non_final_mask = torch.tensor([not is_over(
            next_state) for next_state in batch.next_state], device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack(
            [next_state for next_state in batch.next_state if not is_over(next_state)])
        state_batch = torch.stack([state for state in batch.state])

        action_batch_1 = None
        # action_batch_2 = None

        multiple_outputs = action == Action.CHOOSE_ATTACK_BASE or action == Action.CHOOSE_FORTIFY_SOURCE or action == Action.PLACE
        numeric_output = action == Action.DEFEND or action == Action.CAPTURE

        action_ints = [get_action_int(action) for action in batch.action]
        action_batch_1 = torch.tensor(
            action_ints, device=DEVICE)

        """
        if multiple_outputs:
            action_batch_1 = torch.tensor(
                [get_action_int(action) for action in batch.action], device=DEVICE)
            
            action_batch_2 = torch.tensor(
                [army_proportion for _, army_proportion in batch.action], device=DEVICE)
            
        elif action == Action.CLAIM or action == Action.CHOOSE_ATTACK_TARGET or action == Action.CHOOSE_FORTIFY_TARGET:
            action_batch_1 = torch.tensor(
                [board.territory_to_index[territory] for territory in batch.action], device=DEVICE)
        elif action == Action.CAPTURE or action == Action.DEFEND or action == action.CARDS:
            action_batch_1 = torch.tensor(
                [proportion for proportion in batch.action], device=DEVICE)
        """

        reward_batch = torch.tensor(
            batch.reward, device=DEVICE, dtype=torch.double)

        state_action_values_1 = None
        # state_action_values_2 = None

        if multiple_outputs:
            state_action_values_1, _ = self.network(
                state_batch, action, batch.selected_options)

            state_action_values_1 = state_action_values_1.gather(
                1, action_batch_1.unsqueeze(0))
        else:
            out = self.network(
                state_batch, action, batch.selected_options)
            if numeric_output:
                state_action_values_1 = (
                    out * action_batch_1.unsqueeze(1)).transpose(0, 1)
            else:
                state_action_values_1 = out.gather(
                    1, action_batch_1.unsqueeze(0))

        next_state_values_1 = torch.zeros(
            BATCH_SIZE, device=DEVICE, dtype=torch.double)
        next_state_values_2 = torch.zeros(
            BATCH_SIZE, device=DEVICE, dtype=torch.double)

        if multiple_outputs:
            next_state_values_1, next_state_values_2 = self.target_network(
                non_final_next_states, action, batch.selected_options)

            next_state_values_1 = next_state_values_1.max(1).values.detach()[
                non_final_mask]

            """
            next_state_values_2[non_final_mask] = next_state_values_2.max(1)[
                0].detach()
            """
        else:
            next_state_values_1[non_final_mask] = self.target_network(
                non_final_next_states, action, batch.selected_options).max(1).values.detach()

        expected_state_action_values_1 = (
            next_state_values_1 * GAMMA) + reward_batch
        if expected_state_action_values_1.unsqueeze(0).shape != state_action_values_1.shape:
            raise RuntimeError(
                f'Mismatched shape: {state_action_values_1.shape}, {expected_state_action_values_1.unsqueeze(0).shape}\n{state_action_values_1}\n{expected_state_action_values_1.unsqueeze(0)}')
        loss = self.criterion(state_action_values_1,
                              expected_state_action_values_1.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        """
        if multiple_outputs:
            expected_state_action_values_2 = (
                next_state_values_2 * GAMMA) + reward_batch
            loss = self.criterion(state_action_values_2,
                                  expected_state_action_values_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        """

    def train_step(self, action: Action, board: Board, state, selected_options: list[int]):
        if not self.last_state:
            return

        reward = self.get_reward(
            self.last_state.board, self.last_action, state, board)
        self.add_memory(self.last_risk_action, self.last_state.state, self.last_action,
                        state, reward, self.last_selected_options, board.is_win(self))
        if (self.last_risk_action == Action.CHOOSE_ATTACK_TARGET or self.last_risk_action == Action.CHOOSE_FORTIFY_TARGET) and len(self.last_selected_options) != 43:
            raise RuntimeError(f'{self.last_risk_action} is not 43')

        if len(self.action_memory[self.last_risk_action]) > BATCH_SIZE:
            self.optimize(self.last_risk_action, board)

        if self.steps % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def __init__(self, name: str, network: RiskNet):
        self.name = name
        self.network = network.to(DEVICE)
        if exists(WEIGHTS_PATH):
            print('Loading weights...')
            load_start = time.time()
            self.network.load_state_dict(torch.load(WEIGHTS_PATH))
            load_end = time.time()
            print(f'Weights loaded in {load_end - load_start}')

        self.territories = dict()
        self.hand = list()
        self.steps = 0

        if TRAINING:
            self.target_network = RiskNet().to(DEVICE)
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_network.eval()

            self.optimizer = torch.optim.AdamW(self.network.parameters())

            self.action_memory = {action: deque(
                [], maxlen=MEM_SIZE) for action in Action}
            if exists(MEMORY_PATH):
                print('Loading memory...')
                load_start = time.time()
                with open(MEMORY_PATH, 'rb') as memory:
                    self.memory = pickle.load(memory)
                load_end = time.time()
                print(f'Memory loaded in {load_end - load_start}')

            self.criterion = torch.nn.SmoothL1Loss()
            self.last_state = None
            self.last_action = None
            self.last_risk_action = None
            self.last_selected_options = None
            self.eps = EPS_START

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        state = board.get_state_for_player(
            self, Action.CLAIM, 0, *free_territories)
        valid_selections = self.get_territories_mask(board, free_territories)

        if TRAINING and self.steps > 0:
            self.train_step(Action.CLAIM, board, state, valid_selections)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                claim = RandomPlayer.get_claim(self, board, free_territories)
                self.last_risk_action = Action.CLAIM
                self.last_action = claim
                self.last_selected_options = valid_selections
                self.steps += 1
                return claim

        claim_index = int(torch.argmax(
            self.network(state, Action.CLAIM, valid_selections)).item())
        claim = self.get_territory_from_index(board, claim_index)

        self.steps += 1
        self.last_risk_action = Action.CLAIM
        self.last_action = claim
        self.last_selected_options = valid_selections

        return claim

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        state = board.get_state_for_player(self, Action.PLACE, armies_to_place)
        valid_selections = self.get_territories_mask(
            board, list(self.territories.keys()))

        if TRAINING and self.steps > 0:
            self.train_step(Action.PLACE, board, state, valid_selections)

            self.last_state = LastState(board, state)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                placement = RandomPlayer.place_armies(
                    self, board, armies_to_place)
                self.last_action = placement
                self.last_risk_action = Action.PLACE
                self.last_selected_options = valid_selections
                self.steps += 1
                return placement

        territory_activations, armies_proportion = self.network(
            state, Action.PLACE, valid_selections)
        territory = self.get_territory_from_index(
            board, int(torch.argmax(territory_activations).item()))

        armies = int(torch.floor(armies_proportion *
                     (armies_to_place - 1)).item() + 1)
        self.last_action = (territory, armies)
        self.last_risk_action = Action.PLACE
        self.last_selected_options = valid_selections

        self.steps += 1

        return territory, armies

    def attack(self, board: Board) -> tuple[Territory | None, Territory, int]:
        pre_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_ATTACK_TARGET)

        random_selection = False
        target = None
        base = None
        armies = None

        valid_selections = self.get_territories_mask(
            board, self.get_valid_attack_targets(board, set(list(self.territories.keys()))))
        valid_selections.append(True)

        if TRAINING and self.steps > 0:
            self.train_step(Action.CHOOSE_ATTACK_TARGET,
                            board, pre_selection_state, valid_selections)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                random_selection = True
                target, base, armies = RandomPlayer.attack(self, board)

                self.last_risk_action = Action.CHOOSE_ATTACK_TARGET
                self.last_selected_options = valid_selections
                if target is None:
                    self.last_action = len(board.territories)
                    self.steps += 1
                    return None, None, None
                else:
                    self.last_action = board.territory_to_index[target]

        if not random_selection:
            target_index = int(torch.argmax(
                self.network(pre_selection_state, Action.CHOOSE_ATTACK_TARGET, valid_selections)).item())

            if target_index == len(board.territories):
                self.last_action = target_index
                self.last_risk_action = Action.CHOOSE_ATTACK_TARGET
                self.last_selected_options = valid_selections
                self.steps += 1
                return None, None, None

            target = self.get_territory_from_index(board, target_index)
            self.last_action = target
            self.last_risk_action = Action.CHOOSE_ATTACK_TARGET
            self.last_selected_options = valid_selections

        post_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_ATTACK_BASE, 0, target)
        valid_selections = self.get_territories_mask(
            board, self.get_valid_bases(board, target, set(list(self.territories.keys()))))

        if TRAINING and self.steps > 0:
            self.train_step(Action.CHOOSE_ATTACK_BASE,
                            board, post_selection_state, valid_selections)

            self.last_state = LastState(board, post_selection_state)

        if not random_selection:
            base_activations, armies_activations = self.network(
                post_selection_state, Action.CHOOSE_ATTACK_BASE, valid_selections)
            base = self.get_territory_from_index(
                board, int(torch.argmax(base_activations).item()))

            armies_mask = torch.tensor([1.0 if board.armies[base] > potential_armies else 0.0 for potential_armies in [
                1, 2, 3]], dtype=torch.double).to(DEVICE)
            armies = int(torch.argmax(
                armies_mask * armies_activations).item()) + 1

        self.last_action = (base, armies)
        self.last_risk_action = Action.CHOOSE_ATTACK_BASE
        self.last_selected_options = valid_selections

        self.steps += 1

        return target, base, armies

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        state = board.get_state_for_player(
            self, Action.CAPTURE, attacking_armies, target, base)

        if TRAINING and self.steps > 0:
            self.train_step(Action.CAPTURE, board, state, None)

            self.last_state = LastState(board, state)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                armies = RandomPlayer.capture(
                    self, board, target, base, attacking_armies)
                self.last_action = armies
                self.last_risk_action = Action.CAPTURE
                self.last_selected_options = None
                self.steps += 1
                return armies

        armies = int(attacking_armies + torch.floor(self.network(state,
                     Action.CAPTURE) * (board.armies[base] - attacking_armies - 1)).item())
        self.last_action = armies
        self.last_risk_action = Action.CAPTURE
        self.last_selected_options = None
        self.steps += 1
        return armies

    def defend(self, board: Board, target: Territory, attacking_armies: int) -> int:
        state = board.get_state_for_player(
            self, Action.DEFEND, attacking_armies, target)

        if TRAINING and self.steps > 0:
            self.train_step(Action.DEFEND, board, state, None)

            self.last_state = LastState(board, state)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                armies = RandomPlayer.defend(
                    self, board, target, attacking_armies)
                self.last_action = armies
                self.last_risk_action = Action.DEFEND
                self.steps += 1
                return armies

        if board.armies[target] == 1:
            self.last_action = 1
            self.last_risk_action = Action.DEFEND
            self.last_selected_options = None
            self.steps += 1
            return 1

        defend_activation = self.network(state, Action.DEFEND)

        self.steps += 1

        if defend_activation > 0.5:
            self.last_action = 2
            self.last_risk_action = Action.DEFEND
            self.last_selected_options = None
            return 2
        else:
            self.last_action = 1
            self.last_risk_action = Action.DEFEND
            self.last_selected_options = None
            return 1

    def fortify(self, board: Board) -> tuple[Territory | None, Territory, int]:
        pre_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_FORTIFY_TARGET)

        random_selection = False
        destination = None
        source = None
        armies = None

        valid_destinations = [territory for territory, neighbors in board.territories.items()
                              if territory in self.territories
                              and len([neighbor for neighbor in neighbors if neighbor in self.territories and board.armies[neighbor] > 1]) != 0]

        valid_selections = self.get_territories_mask(
            board, valid_destinations)
        valid_selections.append(1)

        if TRAINING and self.steps > 0:
            self.train_step(Action.CHOOSE_FORTIFY_TARGET,
                            board, pre_selection_state, valid_selections)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                random_selection = True
                destination, source, armies = RandomPlayer.fortify(self, board)
                self.last_risk_action = Action.CHOOSE_FORTIFY_TARGET
                self.last_selected_options = valid_selections
                if destination is None:
                    self.last_action = len(board.territories)
                    self.steps += 1
                    return None, None, None
                else:
                    self.last_action = board.territory_to_index[destination]

        if not random_selection:
            destination_index = int(torch.argmax(self.network(
                pre_selection_state, Action.CHOOSE_FORTIFY_TARGET, valid_selections)).item())

            if destination_index == len(board.territories):
                self.last_action = destination_index
                self.last_risk_action = Action.CHOOSE_FORTIFY_TARGET
                self.last_selected_options = valid_selections
                self.steps += 1
                return None, None, None

            destination = self.get_territory_from_index(
                board, destination_index)
            self.last_action = destination
            self.last_risk_action = Action.CHOOSE_FORTIFY_TARGET
            self.last_selected_options = valid_selections

        post_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_FORTIFY_SOURCE, 0, destination)

        valid_sources = self.get_territories_mask(board, self.get_valid_bases(
            board, destination, set(list(self.territories.keys()))))

        if TRAINING and self.steps > 0:
            self.train_step(Action.CHOOSE_FORTIFY_SOURCE,
                            board, post_selection_state, valid_sources)

            self.last_state = LastState(board, post_selection_state)

        if not random_selection:

            source_index, armies_proportion = self.network(
                post_selection_state, Action.CHOOSE_FORTIFY_SOURCE, valid_sources)

            source = self.get_territory_from_index(
                board, int(torch.argmax(source_index).item()))
            armies = int(torch.floor(1 + armies_proportion *
                                     (board.armies[source] - 2)).item())

        self.last_action = (source, armies)
        self.last_risk_action = Action.CHOOSE_FORTIFY_SOURCE
        self.last_selected_options = valid_sources

        self.steps += 1

        return destination, source, armies

    def use_cards(self, board: Board, matches: list[Card]) -> tuple[Card, Card, Card] | None:
        state = board.get_state_for_player(self, Action.CARDS)

        if TRAINING and self.steps > 0:
            self.train_step(Action.CARDS, board, state, None)

            self.last_state = LastState(board, state)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                cards = RandomPlayer.use_cards(self, board, matches)
                if cards is None:
                    self.last_action = 0
                else:
                    self.last_action = 0
                self.last_risk_action = Action.CARDS
                self.last_selected_options = None
                self.steps += 1
                return cards

        card_proportion = self.network(state, Action.CAPTURE)

        card_choice = None

        if len(self.hand) < 5:
            card_choice = int(torch.floor(
                card_proportion * len(matches)).item())
        else:
            card_choice = int(torch.floor(
                card_proportion * (len(matches) - 1)).item())

        if card_choice == len(matches):
            self.steps += 1
            self.last_action = 0
            self.last_risk_action = Action.CARDS
            self.last_selected_options = None
            self.steps += 1
            return None

        cards = matches[card_choice]
        self.last_action = 0
        self.last_risk_action = Action.CARDS
        self.last_selected_options = None
        self.steps += 1
        return cards

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        state = board.get_state_for_player(self, Action.PLACE, 2)
        valid_selections = self.get_territories_mask(
            board, potential_territories)

        if TRAINING and self.steps > 0:
            self.train_step(Action.PLACE, board, state, valid_selections)

            self.last_state = LastState(board, state)

            if self.eps != EPS_END:
                self.eps -= EPS_DECAY

            if rng.random() <= self.eps:
                choice = RandomPlayer.choose_extra_deployment(
                    self, board, potential_territories)
                self.last_action = choice
                self.last_risk_action = Action.PLACE
                self.last_selected_options = valid_selections
                self.steps += 1
                return choice

        territory_activations, _ = self.network(
            state, Action.PLACE, valid_selections)

        choice = self.get_territory_from_index(
            board, int(torch.argmax(territory_activations).item()))
        self.last_action = choice
        self.last_risk_action = Action.PLACE
        self.last_selected_options = valid_selections
        self.steps += 1
        return choice

    def add_memory(self, action, *args):
        self.action_memory[action].append(Transition(*args))

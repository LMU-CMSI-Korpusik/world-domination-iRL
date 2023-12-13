"""
Different types of Players that can play in Risk. RiskPlayer implemented
while referencing pac-cap final project for Dr. Forney's CMSI 4320

Author: Kieran Ahn
Date: 11/27/2023
"""
from validators import *
from riskGame import Card, Territory, Action
from riskLogic import Board, Player, PlayerState, Rules
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
    'Transition', ('state', 'action', 'next_state', 'reward', 'selected_options', 'is_win'))
LastState = namedtuple('LastState', ('board', 'state'))


class RandomPlayer(Player):
    """
    A very simple Player that just makes random choices in all situations.
    """

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        return rng.choice(list(free_territories))

    def place_armies(self, state: PlayerState, armies_to_place: int) -> tuple[Territory, int]:
        armies_placed = None
        if armies_to_place == 1:
            armies_placed = 1
        else:
            armies_placed = int(rng.integers(
                1, armies_to_place, endpoint=True))
        return (rng.choice(list(self.territories)), armies_placed)

    def attack(self, state: PlayerState) -> tuple[Territory, Territory, int]:
        stop_attack = rng.random() * (self.occupied_territories() + 1) < 1
        occupied_territories = self.territories

        if stop_attack:
            return (None, None, None)

        valid_attack_targets = self.get_valid_attack_targets(
            state, occupied_territories)

        if len(valid_attack_targets) == 0:
            return (None, None, None)

        target = rng.choice(list(valid_attack_targets))

        valid_bases = self.get_valid_bases(
            state, target, occupied_territories)

        base = rng.choice(list(valid_bases))

        attacking_armies = rng.integers(
            1, min(state.armies[base] - 1, 3), endpoint=True)

        return target, base, int(attacking_armies)

    def capture(self, state: PlayerState, target: Territory, base: Territory, attacking_armies: int) -> int:
        if state.armies[base] - 1 == attacking_armies:
            return attacking_armies
        return int(rng.integers(attacking_armies, state.armies[base]))

    def defend(self, state: PlayerState, target: Territory, attacking_armies: int) -> int:
        if state.armies[target] == 1:
            return 1
        else:
            return int(rng.integers(1, 2, endpoint=True))

    def fortify(self, state: PlayerState) -> tuple[Territory, Territory, int]:
        no_fortify = rng.random() * (self.occupied_territories() + 1) < 1

        if no_fortify:
            return (None, None, None)

        occupied_territories = self.territories
        possible_destinations = [territory for territory, neighbors in state.territories.items()
                                 if territory in self.territories
                                 and len([neighbor for neighbor in neighbors if neighbor in self.territories and state.armies[neighbor] > 1]) != 0]

        if len(possible_destinations) == 0:
            return (None, None, None)

        destination = rng.choice(possible_destinations)
        valid_sources = self.get_valid_bases(
            state, destination, occupied_territories)

        source = rng.choice(list(valid_sources))

        fortifying_armies = None

        if state.armies[source] == 2:
            fortifying_armies = 1
        else:
            fortifying_armies = int(rng.integers(1, state.armies[source]))

        return destination, source, fortifying_armies

    def use_cards(self, state: PlayerState, matches: list[Card]) -> tuple[Card, Card, Card]:
        if len(matches) == 0:
            return None
        return rng.choice(matches)

    def choose_extra_deployment(self, state: PlayerState, potential_territories: list[Territory]) -> Territory:
        return rng.choice(potential_territories)


class RiskPlayer(Player):
    """
    A player that implements RiskNet for decisionmaking.
    """

    def __init__(self, name: str, network: RiskNet, memory: dict[Action, deque] = None):
        self.name = name
        self.network = network.to(DEVICE)
        if exists(WEIGHTS_PATH):
            print('Loading weights...')
            load_start = time.time()
            self.network.load_state_dict(torch.load(WEIGHTS_PATH))
            load_end = time.time()
            print(f'Weights loaded in {load_end - load_start}')

        self.territories = set()
        self.hand = list()
        self.steps = 0

        if TRAINING:
            self.target_network = RiskNet().to(DEVICE)
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_network.eval()

            self.optimizer = torch.optim.AdamW(self.network.parameters())

            if not memory:
                self.action_memory = {action: deque(
                    [], maxlen=MEM_SIZE) for action in Action}
                if exists(MEMORY_PATH):
                    print('Loading memory...')
                    load_start = time.time()
                    with open(MEMORY_PATH, 'rb') as memory:
                        self.memory = pickle.load(memory)
                    load_end = time.time()
                    print(f'Memory loaded in {load_end - load_start}')
            else:
                self.action_memory = memory

            self.criterion = torch.nn.SmoothL1Loss()
            self.last_state = None
            self.last_action = None
            self.last_risk_action = None
            self.last_valid_selections = None
            self.action_eps = {action: EPS_START for action in Action}

    @staticmethod
    def get_territories_mask(state: PlayerState, valid_territories: list[Territory] | set[Territory]) -> list[int]:
        """
        Returns a one-hot encoded list of all territories represented in 
        valid_territories, where 1 indicates their inclusion in the collection
        and 0 indicates their exclusion

        :params:\n
        state               --  the state of the game\n
        valid_territories   --  the territories to be included in the mask
        """
        mask = [0 for _ in range(len(state.territories))]
        for territory in valid_territories:
            mask[state.territory_to_index[territory]] = 1
        return mask

    @staticmethod
    def get_territory_from_index(state: PlayerState, territory_index: int) -> Territory:
        """
        Gets a Territory given its index in the sparse row

        :params:\n
        state           --  the state of the game
        territory_index --  the Territory's index in the sparse row
        """
        for territory, index in state.territory_to_index.items():
            if territory_index == index:
                return territory

    def get_reward(self, last_state: PlayerState, last_action, current_state: PlayerState) -> int:
        """
        Gets the reward for a given (state, action) -> next state episode

        :params:\n
        last_state      --  the last observation\n
        last_action     --  the last action the Player took\n
        current_state   --  the current observation
        """
        if current_state.is_winner:
            return 10

        reward = 0

        match(current_state.action):
            case Action.CHOOSE_ATTACK_TARGET:
                if last_action != None:
                    target = last_action

                    reward += 0.01

                    bases_powers = [last_state.armies[base] / (last_state.armies[base] + last_state.armies[target])
                                    for base in self.get_valid_bases(last_state, target, last_state.owned_territories)]
                    # Attacking an opponent with more or equal armies on a territory than all your surrounding territories is not good
                    if all([power <= 0.5 for power in bases_powers]):
                        reward -= 0.02
                else:
                    # If it's not attacking without a good reason, that's bad. Games should be as short as possible.
                    possible_targets = self.get_valid_attack_targets(
                        last_state, last_state.owned_territories)
                    good_bases = len({base for target in possible_targets
                                      for base in self.get_valid_bases(
                                          last_state, target, last_state.owned_territories)
                                      if last_state.armies[base] > 2*last_state.armies[target] + 1})
                    reward -= 0.005 + 0.01 * good_bases

            case Action.CHOOSE_ATTACK_BASE:
                valid_sources = self.get_valid_bases(
                    last_state, last_state.target_territory, last_state.owned_territories)
                source = last_action
                target = last_state.target_territory

                # Attacking with the strongest army you have is always a good option
                other_bases_strengths = [last_state.armies[source] >= last_state.armies[other_base]
                                         for other_base in valid_sources if other_base is not source]

                if len(other_bases_strengths) == 0 or all([last_state.armies[source] >= last_state.armies[other_base] for other_base in valid_sources if other_base is not source]):
                    reward += 0.01

                power_proportion = last_state.armies[source] / \
                    (last_state.armies[target] + last_state.armies[source])
                if power_proportion > 0.5:
                    reward += 0.01 * power_proportion
                else:
                    reward -= 0.01 * (1 - power_proportion)

            case Action.CHOOSE_ATTACK_ARMIES:
                # You always want to be attacking at the maximum number of armies
                reward += 0.01 * last_action

                if last_state.armies[last_state.source_territory] > last_action + 1:
                    reward -= 0.01

                if last_action < 3 and last_state.armies[last_state.target_territory] >= 1:
                    reward -= 0.015

            case Action.DEFEND:
                reward += 0.01 * last_action

            case Action.CLAIM:
                # Securing early borders is a good strategy
                claimed_territory = self.get_territory_from_index(
                    last_state, last_action)
                security_modifier = 0.01
                for territory, neighbors in last_state.territories.items():
                    if territory in last_state.owned_territories and claimed_territory in neighbors:
                        reward += security_modifier

            case Action.PLACE:
                placement = current_state.target_territory
                if last_state.territories[placement].issubset(last_state.owned_territories):
                    # You should not place your armies on territories that have no enemy neighbors
                    reward -= 0.05 * last_action[1]
                else:
                    # The more enemies that surround a territory, the stronger it should be to ward off attacks,
                    # but armies should be split evenly around each border territory
                    source_surrounding_enemies = [
                        neighbor for neighbor in last_state.territories[placement] if neighbor not in last_state.owned_territories]
                    source_surrounding_enemies_strength_modifier = sum(
                        [last_state.armies[enemy_territory] for enemy_territory in source_surrounding_enemies]) / last_state.armies[placement]

                    danger_modifier = 0.01 * len(source_surrounding_enemies) * \
                        source_surrounding_enemies_strength_modifier

                    border_territories = {territory for territory in last_state.owned_territories if any(
                        [neighbor not in last_state.owned_territories for neighbor in last_state.territories[territory]])}
                    border_territory_armies = [
                        last_state.armies[territory] for territory in border_territories]

                    turtling_modifier = (
                        last_state.armies[placement] / sum(border_territory_armies))

                    reward += danger_modifier * \
                        last_action[1] - 0.01 * \
                        len(border_territories) * turtling_modifier

            case Action.CARDS:
                last_matches = Rules.get_matching_cards(last_state.hand)
                card_choice = None
                if len(last_state.hand) < 5:
                    card_choice = int(np.floor(
                        last_action * len(last_matches)))
                else:
                    card_choice = int(np.floor(
                        last_action * (len(last_matches) - 1)))
                if not len(last_state.hand) == card_choice:
                    # Getting extra armies for turning in cards that have your territories on them is always good
                    if any([card.territory in last_state.territories for match in last_matches for card in match]) and any([card.territory in last_state.territories for card in last_matches[card_choice]]):
                        reward += 0.04

                    # Turning in cards is a good idea
                    if len(last_state.hand) < len(current_state.hand):
                        reward += 0.01

            case Action.CHOOSE_FORTIFY_TARGET:
                if last_action != None:
                    destination = current_state.target_territory
                    source_surrounding_enemies = len(
                        [neighbor for neighbor in last_state.territories[destination] if neighbor not in last_state.owned_territories])
                    if source_surrounding_enemies == 0:
                        # Fortifying a Territory with no enemies next to it is bad, but sometimes necessary
                        reward -= 0.005
                    else:
                        reward += 0.01 * source_surrounding_enemies
                else:
                    # Hoarding armies in one Territory is not a good idea, especially when other Territories need them.
                    good_destinations = len([territory
                                             for enemy_territory in self.get_valid_attack_targets(last_state, last_state.owned_territories)
                                             for territory in last_state.territories[enemy_territory]
                                             if territory in last_state.owned_territories
                                             and any([last_state.armies[base] > 2*last_state.armies[territory] for base in self.get_valid_bases(last_state, territory, last_state.owned_territories)])])
                    reward -= 0.01 * good_destinations + 0.005

            case Action.CHOOSE_FORTIFY_SOURCE:
                source = current_state.source_territory
                valid_sources = self.get_valid_bases(
                    last_state, last_state.target_territory, last_state.owned_territories)
                source_surrounding_enemies = {
                    neighbor for neighbor in last_state.territories[source] if neighbor not in last_state.owned_territories}
                if len(source_surrounding_enemies) > 0:
                    # Taking armies away from a Territory that needs it is not a good idea, but if you're stronger, it's not that bad
                    source_surrounding_enemies_strengths = sum(
                        [last_state.armies[neighbor] for neighbor in source_surrounding_enemies])
                    source_surrounding_enemies_strength_modifier = source_surrounding_enemies_strengths / \
                        (source_surrounding_enemies_strengths +
                         last_state.armies[source])
                    reward -= 0.1 * \
                        len(source_surrounding_enemies) * \
                        source_surrounding_enemies_strength_modifier

                if all([last_state.armies[source] >= last_state.armies[other_source] for other_source in valid_sources if other_source is not source]):
                    # Fortifying with the Territory that can most spare it is good
                    reward += 0.01

            case Action.CHOOSE_FORTIFY_ARMIES | Action.CAPTURE:
                source = last_state.source_territory
                target = last_state.target_territory
                armies_moved_proportion = last_action

                armies_remaining_proportion = 1 - last_action

                source_surrounding_enemies = [
                    neighbor for neighbor in last_state.territories[source] if neighbor not in last_state.owned_territories]
                target_surrounding_enemies = [
                    neighbor for neighbor in last_state.territories[target] if neighbor not in last_state.owned_territories]
                if len(target_surrounding_enemies) == 0:
                    # If there's no enemies aroumd the target, don't move more than you have to.
                    reward -= 0.5 * armies_moved_proportion
                elif len(source_surrounding_enemies) > 0:
                    # Need to balance leaving behind enough armies to defend vs. attacking new territories
                    source_surrounding_enemies_strength = sum(
                        [last_state.armies[enemy_territory] - 1 for enemy_territory in source_surrounding_enemies])

                    source_surrounding_enemies_strength_modifier = source_surrounding_enemies_strength / \
                        (source_surrounding_enemies_strength +
                         last_state.armies[source])

                    source_danger_modifier = 0.1 * \
                        len(source_surrounding_enemies) * \
                        source_surrounding_enemies_strength_modifier

                    target_surrounding_enemies_strengths = sum(
                        [last_state.armies[enemy_territory] - 1 for enemy_territory in target_surrounding_enemies])

                    target_surrounding_enemies_strength_modifier = target_surrounding_enemies_strengths / (target_surrounding_enemies_strengths + (
                        last_state.armies[target] if last_state.action == Action.CHOOSE_FORTIFY_TARGET else last_state.armies_used))

                    target_danger_modifier = 0.1 * \
                        len(target_surrounding_enemies) * \
                        target_surrounding_enemies_strength_modifier

                    reward += target_danger_modifier * armies_moved_proportion
                    reward -= source_danger_modifier * armies_moved_proportion
                else:
                    # ...but if there's no need to defend, move everything.
                    reward += 0.5 * armies_moved_proportion

        return reward

    def optimize(self, action: Action, state: PlayerState):
        """
        Performs an SGD step on the Player's policy network

        :params:\n
        action      --  the turn state we are optimizing\n
        stat        --  any state
        """
        sample = random.sample(self.action_memory[action], BATCH_SIZE)
        batch = Transition(*zip(*sample))

        def get_action_int(to_convert):
            if to_convert is None:
                return 1 if action == Action.CARDS else len(state.territories)
            if action is Action.PLACE:
                return (state.territory_to_index[to_convert[0]], to_convert[1])
            if type(to_convert) is Territory:
                return state.territory_to_index[to_convert]
            if type(to_convert) is int:
                return to_convert
            if type(to_convert) is float:
                return to_convert

        non_final_mask = torch.tensor(
            [not next_state.is_winner for next_state in batch.next_state], device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack(
            [next_state.get_features() for next_state in batch.next_state if not next_state.is_winner])
        state_batch = torch.stack([state.get_features()
                                  for state in batch.state])

        action_batch_1 = None
        action_batch_2 = None

        multiple_outputs = action == Action.PLACE
        numeric_output = action == Action.CAPTURE or action == Action.CARDS or action == Action.CHOOSE_FORTIFY_ARMIES
        no_selectable_options = action == Action.CARDS or action == Action.CAPTURE or action == Action.CHOOSE_FORTIFY_ARMIES

        action_ints = [get_action_int(action) for action in batch.action]

        if multiple_outputs:
            action_ints_1, action_ints_2 = zip(*action_ints)
            action_batch_1 = torch.tensor(
                action_ints_1, device=DEVICE
            )
            action_batch_2 = torch.tensor(
                action_ints_2, device=DEVICE
            )
        else:
            action_batch_1 = torch.tensor(
                action_ints, device=DEVICE)

        reward_batch = torch.tensor(
            batch.reward, device=DEVICE, dtype=torch.double)

        state_action_values_1 = None
        state_action_values_2 = None

        if multiple_outputs:
            state_action_values_1, state_action_values_2 = self.network(
                state_batch, action, batch.selected_options)

            state_action_values_1 = state_action_values_1.gather(
                1, action_batch_1.unsqueeze(0))

            state_action_values_2 = (
                state_action_values_2 * action_batch_2.unsqueeze(1)).transpose(0, 1)
        else:
            out = self.network(
                state_batch, action, None if no_selectable_options else batch.selected_options)
            if numeric_output:
                state_action_values_1 = (
                    out * action_batch_1.unsqueeze(1)).transpose(0, 1)
            else:
                if action == Action.CHOOSE_ATTACK_ARMIES or action == Action.DEFEND:
                    action_batch_1 = action_batch_1 - 1
                state_action_values_1 = out.gather(
                    1, action_batch_1.unsqueeze(0))

        next_state_values_1 = torch.zeros(
            BATCH_SIZE, device=DEVICE, dtype=torch.double)
        next_state_values_2 = torch.zeros(
            BATCH_SIZE, device=DEVICE, dtype=torch.double)

        with torch.no_grad():
            if multiple_outputs:
                target_out_1, target_out_2 = self.target_network(
                    non_final_next_states, action, batch.selected_options)

                next_state_values_1[non_final_mask] = target_out_1.max(
                    1).values.detach()

                next_state_values_2[non_final_mask] = target_out_2.max(
                    1).values.detach()

            else:
                next_state_values_1[non_final_mask] = self.target_network(
                    non_final_next_states, action, None if no_selectable_options else batch.selected_options).max(1).values.detach()

        expected_state_action_values_1 = (
            next_state_values_1 * GAMMA) + reward_batch
        if expected_state_action_values_1.unsqueeze(0).shape != state_action_values_1.shape:
            raise RuntimeError(
                f'Mismatched shape: {state_action_values_1.shape}, {expected_state_action_values_1.unsqueeze(0).shape}\n{state_action_values_1}\n{expected_state_action_values_1.unsqueeze(0)}')

        loss = None

        if multiple_outputs:
            expected_state_action_values_2 = (
                next_state_values_2 * GAMMA) + reward_batch
            loss = self.criterion(torch.cat((state_action_values_1, state_action_values_2), 1), torch.cat(
                (expected_state_action_values_1.unsqueeze(0), expected_state_action_values_2.unsqueeze(0)), 1))

        else:
            loss = self.criterion(state_action_values_1,
                                  expected_state_action_values_1.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self, action: Action, state: PlayerState):
        """
        Performs a single training step for the model, which means a single
        optimization step, then a single target update

        :params:\n
        action      --  the current turn phase\n
        state       --  the state of the game
        """
        if not self.last_state:
            return

        if len(self.action_memory[action]) > BATCH_SIZE:
            self.optimize(action, state)

        policy_state_dict = self.network.state_dict()
        target_state_dict = self.target_network.state_dict()
        for parameter in policy_state_dict:
            target_state_dict[parameter] = policy_state_dict[parameter] * \
                TAU + target_state_dict[parameter]*(1 - TAU)
        self.target_network.load_state_dict(target_state_dict)

    def get_claim(self, state: PlayerState, free_territories: set[Territory]) -> Territory:
        valid_selections = self.get_territories_mask(state, free_territories)

        self.last_risk_action = Action.CLAIM
        self.last_valid_selections = valid_selections
        self.last_state = state
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                claim = RandomPlayer.get_claim(self, state, free_territories)
                self.last_action = claim
                return claim

        claim_index = int(torch.argmax(
            self.network(state.get_features(), Action.CLAIM, valid_selections)).item())
        claim = self.get_territory_from_index(state, claim_index)

        self.last_action = claim

        return claim

    def place_armies(self, state: PlayerState, armies_to_place: int) -> tuple[Territory, int]:
        valid_selections = self.get_territories_mask(
            state, list(self.territories))

        self.last_risk_action = Action.PLACE
        self.last_valid_selections = valid_selections
        self.last_state = state
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                armies_proportion = rng.random()
                armies = int(np.round(armies_proportion *
                                      (armies_to_place - 1)) + 1)

                territory = rng.choice(list(self.territories))
                self.last_action = (territory, float(armies_proportion))
                return territory, armies

        territory_activations, armies_proportion = self.network(
            state.get_features(), Action.PLACE, valid_selections)
        territory = self.get_territory_from_index(
            state, int(torch.argmax(territory_activations).item()))

        armies = int(torch.floor(armies_proportion *
                     (armies_to_place - 1)).item() + 1)
        self.last_action = (territory, float(armies_proportion.item()))

        self.steps += 1

        return territory, armies

    def attack(self, state: PlayerState) -> tuple[Territory | None, Territory, int]:
        target = None
        base = None
        armies = None

        target = self.select_attack_target(state)

        target_selected_state = PlayerState(state.territories, state.owned_territories, state.action, state.other_territories, state.armies, state.captured_territory,
                                            state.won_battle, state.source_territory, target, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        self.observe(target_selected_state)

        if target is None:
            return None, None, None

        self.last_state = PlayerState(state.territories, state.owned_territories, state.action, state.other_territories, state.armies, state.captured_territory,
                                      state.won_battle, state.source_territory, target, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        base = self.select_attack_base(self.last_state, target)

        base_selected_state = PlayerState(state.territories, state.owned_territories, Action.CHOOSE_ATTACK_BASE, state.other_territories, state.armies,
                                          state.captured_territory, state.won_battle, base, target, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        self.observe(base_selected_state)

        self.last_state = PlayerState(state.territories, state.owned_territories, state.action, state.other_territories, state.armies, state.captured_territory,
                                      state.won_battle, base, target, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        armies = self.select_attack_armies(self.last_state, base)

        return target, base, armies

    def select_attack_target(self, state: PlayerState) -> Territory | None:
        """
        Selects a Territory to attack, or chooses not to attack

        :params:\n
        state       --  the state of the game

        :returns:\n
        target              --  the targeted Territory or None\n
        random_selection    --  whether or not the target was chosen randomly\n
        random_choices      --  if the target was chosen randomly (i.e., due
        to epsilon-greedy), the random choices made by the RandomPlayer, else
        None
        """
        valid_targets = self.get_valid_attack_targets(state, self.territories)
        valid_selections = self.get_territories_mask(
            state, valid_targets)
        valid_selections.append(1)
        valid_targets.add(None)

        self.last_state = state
        self.last_valid_selections = valid_selections
        self.last_risk_action = Action.CHOOSE_ATTACK_TARGET
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                target = rng.choice(list(valid_targets))

                self.last_action = target

                return target

        target_index = int(torch.argmax(
            self.network(state.get_features(), Action.CHOOSE_ATTACK_TARGET, valid_selections)).item())

        if target_index == len(state.territories):
            self.last_action = None
            return None

        target = self.get_territory_from_index(state, target_index)
        self.last_action = target
        return target

    def select_attack_base(self, state: PlayerState, target: Territory) -> Territory:
        """
        Selects the Territory whose armies the Player will use to attack the 
        targeted Territory

        :params:\n
        state       --  the state of the game\n
        target      --  the Territory to be attacked

        :returns:\n
        base        --  the Territory whose armies will be used in the attack
        """
        valid_bases = self.get_valid_bases(state, target, self.territories)
        valid_selections = self.get_territories_mask(
            state, valid_bases)

        self.last_risk_action = Action.CHOOSE_ATTACK_BASE
        self.last_valid_selections = valid_selections
        self.last_state = state
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                target = rng.choice(list(valid_bases))

                self.last_action = target

                return target

        base_activations = self.network(
            state.get_features(), Action.CHOOSE_ATTACK_BASE, valid_selections)

        base_index = int(torch.argmax(base_activations).item())

        base = self.get_territory_from_index(
            state, base_index)

        self.last_action = base

        return base

    def select_attack_armies(self, state: PlayerState, base: Territory) -> int:
        """
        Selects the number of armies to use when attacking a target Territory

        :params:\n
        state       --  the state of the game\n
        base        --  the Territory from which the Player is attacking

        :returns:
        armies      --  the number of armies to be used in the attack
        """
        valid_selections = [1 if state.armies[base] > potential_armies else 0 for potential_armies in [
            1, 2, 3]]
        valid_armies = [armies for armies in [
            1, 2, 3] if state.armies[base] > armies]

        self.last_risk_action = Action.CHOOSE_ATTACK_ARMIES
        self.last_valid_selections = valid_selections
        self.last_state = state
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                armies = int(rng.choice(valid_armies))

                self.last_action = armies

                return armies

        armies_activations = self.network(
            state.get_features(), Action.CHOOSE_ATTACK_ARMIES, valid_selections)
        armies = int(torch.argmax(armies_activations).item()) + 1

        self.last_action = armies

        return armies

    def capture(self, state: PlayerState, target: Territory, base: Territory, attacking_armies: int) -> int:
        self.last_risk_action = Action.CAPTURE
        self.last_valid_selections = None
        self.last_state = state
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                armies_proportion = rng.random()

                armies = int(attacking_armies + np.round(armies_proportion *
                             (state.armies[base] - attacking_armies - 1)))

                self.last_action = float(armies_proportion)
                return armies

        armies_proportion = self.network(state.get_features(), Action.CAPTURE)
        self.last_action = float(armies_proportion.item())

        armies = int(attacking_armies + torch.floor(armies_proportion *
                     (state.armies[base] - attacking_armies - 1)).item())
        return armies

    def defend(self, state: PlayerState, target: Territory, attacking_armies: int) -> int:
        self.last_risk_action = Action.DEFEND
        self.last_state = state
        self.steps += 1
        self.last_valid_selections = [
            1.0 if state.armies[target] >= armies else 0.0 for armies in [1, 2]]

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                armies = RandomPlayer.defend(
                    self, state, target, attacking_armies)
                self.last_action = armies
                return armies

        if state.armies[target] == 1:
            self.last_action = 1
            return 1

        defend_armies = int(torch.argmax(self.network(
            state.get_features(), Action.DEFEND, self.last_valid_selections)).item()) + 1

        self.last_action = defend_armies
        return defend_armies

    def fortify(self, state: PlayerState) -> tuple[Territory | None, Territory, int]:
        destination = None
        source = None
        armies = None

        destination = self.select_fortify_destination(state)

        destination_selected_state = PlayerState(state.territories, state.owned_territories, state.action, state.other_territories, state.armies, state.captured_territory,
                                                 state.won_battle, state.source_territory, destination, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        self.observe(destination_selected_state)

        if destination is None:
            return None, None, None

        self.last_state = PlayerState(state.territories, state.owned_territories, state.action, state.other_territories, state.armies, state.captured_territory,
                                      state.won_battle, state.source_territory, destination, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        source = self.select_fortify_source(self.last_state, destination)

        source_selected_state = PlayerState(state.territories, state.owned_territories, Action.CHOOSE_FORTIFY_SOURCE, state.other_territories, state.armies,
                                            state.captured_territory, state.won_battle, source, destination, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        self.observe(source_selected_state)

        self.last_state = PlayerState(state.territories, state.owned_territories, state.action, state.other_territories, state.armies, state.captured_territory,
                                      state.won_battle, source, destination, state.armies_used, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        armies = self.select_fortify_armies(self.last_state, source)

        armies_selected_state = PlayerState(state.territories, state.owned_territories, Action.CHOOSE_FORTIFY_ARMIES, state.other_territories, state.armies,
                                            state.captured_territory, state.won_battle, source, destination, armies, state.hand, state.matches_traded, state.territory_to_index, state.is_winner)

        self.observe(armies_selected_state)

        return destination, source, armies

    def select_fortify_destination(self, state: PlayerState) -> Territory | None:
        """
        Selects a Territory to fortify, or chooses not to fortify

        :params:\n
        state       --  the state of the game

        :returns:\n
        destination         --  the Territory to be fortified or None\n
        random_selection    --  whether or not the destination was selected
        randomly\n
        random_choices      --  if the destination was chosen randomly (i.e., 
        due to epsilon-greedy), the random choices made by the RandomPlayer, 
        else None
        """
        valid_destinations = [territory for territory, neighbors in state.territories.items()
                              if territory in self.territories
                              and len([neighbor for neighbor in neighbors if neighbor in self.territories and state.armies[neighbor] > 1]) != 0]

        valid_selections = self.get_territories_mask(
            state, valid_destinations)
        valid_selections.append(1)
        valid_destinations.append(None)

        self.last_state = state
        self.last_risk_action = Action.CHOOSE_FORTIFY_TARGET
        self.last_valid_selections = valid_selections
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                destination = rng.choice(list(valid_destinations))

                self.last_action = destination

                return destination

        destination_index = int(torch.argmax(self.network(
            state.get_features(), Action.CHOOSE_FORTIFY_TARGET, valid_selections)).item())

        if destination_index == len(state.territories):
            self.last_action = None
            return None

        destination = self.get_territory_from_index(
            state, destination_index)

        self.last_action = destination

        return destination

    def select_fortify_source(self, state: PlayerState, destination: Territory) -> Territory:
        """
        Selects the Territory whose armies will be used to reinforce another 
        Territory

        :params:\n
        state       --  the state of the game\n
        destination --  the Territory which will be fortified

        :returns:
        source      --  the Territory whose armies will be used
        """
        valid_bases = self.get_valid_bases(
            state, destination, self.territories)
        valid_sources = self.get_territories_mask(state, valid_bases)

        self.last_risk_action = Action.CHOOSE_FORTIFY_SOURCE
        self.last_valid_selections = valid_sources
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                source = rng.choice(list(valid_bases))

                self.last_action = source

                return source

        source_index = self.network(
            state.get_features(), Action.CHOOSE_FORTIFY_SOURCE, valid_sources)

        source = self.get_territory_from_index(
            state, int(torch.argmax(source_index).item()))

        self.last_action = source

        return source

    def select_fortify_armies(self, state: PlayerState, source: Territory) -> int:
        """
        Selects the number of armies from a territory to use in fortification

        :params:\n
        state       --  the state of the game\n
        source      --  the Territory from which the Player is taking the
        armies

        :returns:\n
        armies      --  the number of armies to take from the source and put
        into the destination
        """
        self.last_state = state
        self.last_risk_action = Action.CHOOSE_FORTIFY_ARMIES
        self.last_valid_selections = None
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                armies_proportion = rng.random()
                armies = int(np.round(1 + armies_proportion *
                                      (state.armies[source] - 2)))

                self.last_action = float(armies_proportion)

                return armies

        armies_proportion = self.network(
            state.get_features(), Action.CHOOSE_FORTIFY_ARMIES)

        armies = int(torch.floor(1 + armies_proportion *
                                 (state.armies[source] - 2)).item())

        self.last_action = float(armies_proportion.item())

        return armies

    def use_cards(self, state: PlayerState, matches: list[tuple[Card, Card, Card]]) -> tuple[Card, Card, Card] | None:
        self.last_risk_action = Action.CARDS
        self.last_valid_selections = None
        self.steps += 1

        card_choice = None

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                cards = RandomPlayer.use_cards(self, state, matches)
                if cards is None:
                    self.last_action = 1
                else:
                    card_proportion = rng.random()

                    if len(self.hand) < 5:
                        card_choice = int(np.round(
                            card_proportion * len(matches)))
                    else:
                        card_choice = int(np.round(
                            card_proportion * (len(matches) - 1)))

                    self.last_action = float(card_proportion)

                    if card_choice == len(matches):
                        return None

                    cards = matches[card_choice]

                return cards

        card_proportion = self.network(state.get_features(), Action.CAPTURE)
        self.last_action = card_proportion.item()

        if len(self.hand) < 5:
            card_choice = int(torch.floor(
                card_proportion * len(matches)).item())
        else:
            card_choice = int(torch.floor(
                card_proportion * (len(matches) - 1)).item())

        if card_choice == len(matches):
            return None

        cards = matches[card_choice]

        return cards

    def choose_extra_deployment(self, state: PlayerState, potential_territories: list[Territory]) -> Territory:
        valid_selections = self.get_territories_mask(
            state, potential_territories)

        self.last_risk_action = Action.PLACE
        self.last_valid_selections = valid_selections
        self.steps += 1

        if TRAINING:
            if self.action_eps[self.last_risk_action] != EPS_END:
                self.action_eps[self.last_risk_action] -= EPS_DECAY

            if rng.random() <= self.action_eps[self.last_risk_action]:
                choice = RandomPlayer.choose_extra_deployment(
                    self, state, potential_territories)
                self.last_action = (choice, 2)

                return choice

        territory_activations, _ = self.network(
            state.get_features(), Action.PLACE, valid_selections)

        choice = self.get_territory_from_index(
            state, int(torch.argmax(territory_activations).item()))
        self.last_action = (choice, 2)
        return choice

    def add_memory(self, action, *args):
        """
        Adds a memory to the replay buffer

        :params:\n
        action      --  The action which was just performed
        *args       --  The arguments for a Transition tuple
        """
        self.action_memory[action].append(Transition(*args))

    def observe(self, state: PlayerState):
        """
        Observes a state and stores it in the replay memory, then performs an 
        optimization step.

        :params:\n
        state       --  The current state of the board from the Player's view
        """
        if not TRAINING:
            return

        reward = self.get_reward(
            self.last_state, self.last_action, state)

        self.add_memory(self.last_risk_action, self.last_state, self.last_action,
                        state, reward, self.last_valid_selections, state.is_winner)

        if (self.last_risk_action == Action.CHOOSE_ATTACK_TARGET or self.last_risk_action == Action.CHOOSE_FORTIFY_TARGET) and len(self.last_valid_selections) != 43:
            raise RuntimeError(f'{self.last_risk_action} is not 43')

        if len(self.action_memory[self.last_risk_action]) > BATCH_SIZE:
            self.optimize(self.last_risk_action, state)

        if self.steps > 0:
            self.train_step(state.action, state)

    def final(self, state: PlayerState):
        self.last_state = None
        self.last_valid_selections = None
        self.last_risk_action = None
        self.last_action = None
        self.steps = 0

        if TRAINING and state.is_winner:
            print(f'Saving model for player {self.name}...')
            save_init = time.time()
            torch.save(self.network.state_dict(), WEIGHTS_PATH)
            with open(MEMORY_PATH, 'wb') as memory:
                pickle.dump(self.action_memory, memory)
            save_end = time.time()
            print(f'Save completed in {save_end - save_init}')

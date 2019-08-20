from math import ceil
from typing import Dict

from game.board import Harbor, Board
from game.catan_state import CatanState
from game.resource import Resource, ResourceAmounts

from players.abstract_player import AbstractPlayer, Colony
from algorithms.mcts import MCTS, MCTSNode
from collections import Counter
import numpy as np


class MCTSPlayer(AbstractPlayer):
    """
    This class defines a player that chooses best move by using the MCTS algorithm
    """

    def __init__(self, id, seed=None, iterations=1000, exploration_param=1.4):
        assert seed is None or (isinstance(seed, int) and seed > 0)
        super().__init__(id, seed)

        self.iterations = iterations
        self.exploration_param = exploration_param
        self.scores_by_player = None


    def choose_move(self, state: CatanState):
        self.scores_by_player = state.get_scores_by_player_indexed()
        next_moves = state.get_next_moves()
        if len(next_moves) <= 1:
            return next_moves[0]
        if state.is_initialisation_phase():
            best_move, best_score = None, 0
            for move in next_moves:
                state.make_move(move)
                score = self.initialization_phase_heuaristic(state)
                state.unmake_move(move)
                if score > best_score:
                    best_move = move
                    best_score = score
            return best_move
        mcts = MCTS(MCTSNode(state), next_moves, self.exploration_param)
        mcts.do_n_rollouts(self.iterations)
        return mcts.choose().move


    def choose_resources_to_drop(self) -> Dict[Resource, int]:
        if self.in_first_phase():
            return self.drop_resources_in_first_phase()
        return self.drop_resources_in_final_phase()


    def drop_resources_in_first_phase(self):
        resources_count = sum(self.resources.values())
        if resources_count < 8:
            return {}
        resources_to_drop_count = ceil(resources_count / 2)

        resources_for_city = sum(ResourceAmounts.city.values())
        cities_removed = 0
        while self.can_settle_city() and resources_count >= resources_to_drop_count + resources_for_city:
            self.remove_resources_and_piece_for_city()
            resources_count -= resources_for_city
            cities_removed += 1

        resources_for_settlement = sum(ResourceAmounts.settlement.values())
        settlements_removed = 0
        while self.can_settle_settlement() and resources_count >= resources_to_drop_count + resources_for_settlement:
            self.remove_resources_and_piece_for_settlement()
            resources_count -= resources_for_city
            settlements_removed += 1

        resources_for_road = sum(ResourceAmounts.road.values())
        roads_removed = 0
        while self.can_pave_road() and resources_count >= resources_to_drop_count + resources_for_road:
            self.remove_resources_and_piece_for_road()
            resources_count -= resources_for_road
            roads_removed += 1

        resources_for_development_card = sum(ResourceAmounts.development_card.values())
        development_cards_removed = 0
        while self.has_resources_for_development_card() and \
                resources_count >= resources_to_drop_count + resources_for_development_card:
            self.remove_resources_for_development_card()
            resources_count -= resources_for_development_card
            development_cards_removed += 1

        possible_resources_to_drop = [resource for resource, amount in self.resources.items() for i in range(amount)]
        resources_to_drop = Counter(self._random_choice(possible_resources_to_drop, resources_to_drop_count, replace=False))

        for i in range(cities_removed):
            self.add_resources_and_piece_for_city()
        for i in range(settlements_removed):
            self.add_resources_and_piece_for_settlement()
        for i in range(development_cards_removed):
            self.add_resources_for_development_card()
        for i in range(roads_removed):
            self.add_resources_and_piece_for_road()

        return resources_to_drop


    def initialization_phase_heuaristic(self, state, weights=np.ones(10)):
        resource_expectation = self.get_resource_expectation(self, state)
        total_expectation = sum(resource_expectation.values())
        brick_expectation = resource_expectation[Resource.Brick]
        lumber_expectation = resource_expectation[Resource.Lumber]
        wool_expectation = resource_expectation[Resource.Wool]
        grain_expectation = resource_expectation[Resource.Grain]
        ore_expectation = resource_expectation[Resource.Ore]

        # define which probabilities are considered  'decent'. for example: if decent = 4, then 4,5,6,8,9,10 are considered decent
        decent = 4

        has_decent_road_resources = 0
        if brick_expectation >= state.probabilities_by_dice_values[decent] and lumber_expectation >= state.probabilities_by_dice_values[decent]:
            has_decent_road_resources = 1

        has_decent_settlement_resources = 0
        if (brick_expectation >= state.probabilities_by_dice_values[decent]) and lumber_expectation >= state.probabilities_by_dice_values[decent] and wool_expectation >= state.probabilities_by_dice_values[decent] and grain_expectation >= state.probabilities_by_dice_values[decent]:
            has_decent_settlement_resources = 1

        has_decent_city_resources = 0
        if ore_expectation >= state.probabilities_by_dice_values[decent + 1] and grain_expectation >= state.probabilities_by_dice_values[decent]:
            has_decent_city_resources = 1

        has_harbor = 0
        for harbor_type in Harbor:
            if state.board.is_player_on_harbor(self, harbor_type):
                has_harbor += 1

        values = np.zeros(10)
        values[0] = brick_expectation
        values[1] = lumber_expectation
        values[2] = wool_expectation
        values[3] = grain_expectation
        values[4] = ore_expectation
        values[5] = total_expectation
        values[6] = has_decent_road_resources
        values[7] = has_decent_settlement_resources
        values[8] = has_decent_city_resources

        return np.dot(values, weights)

    @staticmethod
    def get_resource_expectation(player, state):
        """
        calculates the expected resource yield per one turn per player.
        :return: a dictionary of the resource expectation of the given player.
        each resource is a key, and it's value is that player's expected yield.
        """
        res_yield = {Colony.Settlement: 1, Colony.City: 2}
        resources = {r: 0 for r in Resource}

        for location in state.board.get_locations_colonised_by_player(player):
            colony_yield = res_yield[state.board.get_colony_type_at_location(location)]
            for land in state.board._roads_and_colonies.node[location][Board.lands]:  # The adjacent lands to the location we check
                if land.resource is None:  # If this is a desert - do nothing
                    continue
                calc = colony_yield * state.probabilities_by_dice_values[land.dice_value]
                resources[land.resource] += calc

        return resources

    def drop_resources_in_final_phase(self):
        resources_count = sum(self.resources.values())
        if resources_count < 8:
            return {}
        resources_to_drop_count = ceil(resources_count / 2)

        resources_for_city = sum(ResourceAmounts.city.values())
        cities_removed = 0
        while self.can_settle_city() and resources_count >= resources_to_drop_count + resources_for_city:
            self.remove_resources_and_piece_for_city()
            resources_count -= resources_for_city
            cities_removed += 1

        resources_for_settlement = sum(ResourceAmounts.settlement.values())
        settlements_removed = 0
        while self.can_settle_settlement() and resources_count >= resources_to_drop_count + resources_for_settlement:
            self.remove_resources_and_piece_for_settlement()
            resources_count -= resources_for_city
            settlements_removed += 1

        resources_for_development_card = sum(ResourceAmounts.development_card.values())
        development_cards_removed = 0
        while self.has_resources_for_development_card() and \
                resources_count >= resources_to_drop_count + resources_for_development_card:
            self.remove_resources_for_development_card()
            resources_count -= resources_for_development_card
            development_cards_removed += 1

        resources_for_road = sum(ResourceAmounts.road.values())
        roads_removed = 0
        while self.can_pave_road() and resources_count >= resources_to_drop_count + resources_for_road:
            self.remove_resources_and_piece_for_road()
            resources_count -= resources_for_road
            roads_removed += 1

        possible_resources_to_drop = [resource for resource, amount in self.resources.items() for i in range(amount)]
        resources_to_drop = Counter(self._random_choice(possible_resources_to_drop, resources_to_drop_count, replace=False))

        for i in range(cities_removed):
            self.add_resources_and_piece_for_city()
        for i in range(settlements_removed):
            self.add_resources_and_piece_for_settlement()
        for i in range(development_cards_removed):
            self.add_resources_for_development_card()
        for i in range(roads_removed):
            self.add_resources_and_piece_for_road()

        return resources_to_drop

    def in_first_phase(self, state=None):
        my_victory_points = self.scores_by_player[self.get_id()]
        return my_victory_points <= 7

    def __str__(self):
        return str(self.__class__.__name__) + "@" + str(self.exploration_param) + "@" + str(self.iterations)


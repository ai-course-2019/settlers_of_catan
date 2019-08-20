from game.catan_state import CatanState
from players.expectimax_baseline_player import ExpectimaxBaselinePlayer
from players.expectimax_weighted_probabilities_with_filter_player import *
from players.abstract_player import *
from game.resource import *
from game.board import *
from game.pieces import *
from game.development_cards import *
from math import *
import numpy as np

import copy
from players.random_player import RandomPlayer
from collections import Counter
from players.filters import *


MAX_ITERATIONS = 10

TOTAL_WEIGHTS = 35

NUM_OF_WEIGHTS =7


class Winner(ExpectimaxBaselinePlayer):

    # default_winning_weights = np.array([0, 45, -1, -0.1, 1, 1,-1])

    default_winning_weights = np.array([0, 45, -1, -0.1, 1, 1,-1, 0, 8, -1, -0.1, 3, 1,-2]) # 7 weights for first phase, 7 weights for last phase

    # winning_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, -0.1, -0.1, -0.1, -0.1, 1, 1])


    def __init__(self, id, seed=None, timeout_seconds=5, weights=default_winning_weights):
        super().__init__(id=id, seed=seed, timeout_seconds=timeout_seconds, heuristic=self.winning_heuristic, filter_moves=self.filter_moves(seed))

        self.scores_by_player = None
        self._players_and_factors = None
        self.winner_initialization_phase_weights = None
        self.winner_first_phase_weights = None
        self.winner_last_phase_weights = None
        self.expectimax_weights = {Colony.City: 2, Colony.Settlement: 1, Road.Paved: 0.4,
                                   DevelopmentCard.VictoryPoint: 1, DevelopmentCard.Knight: 2.0 / 3.0}

        self.initialize_weights(weights)


    def initialize_weights(self, given_weights):
        """
        initializes a np.array of weights for our heuristic.
        the initialization is based on the given weights.
        :param given_weights:
        :return:
        """
        self.winner_first_phase_weights = np.ones(TOTAL_WEIGHTS)
        for i in range(10):
            self.winner_first_phase_weights[i] = given_weights[0]

        for i in range(10, 20):
            self.winner_first_phase_weights[i] = given_weights[1]
        self.winner_first_phase_weights[20] = given_weights[1] / 5

        for i in range(22, 23):
            self.winner_first_phase_weights[i] = given_weights[2]

        for i in range(28, 32):
            self.winner_first_phase_weights[i] = given_weights[3]

        self.winner_first_phase_weights[32] = given_weights[4]

        self.winner_first_phase_weights[33] = given_weights[5]

        self.winner_first_phase_weights[34] = given_weights[6]

        self.winner_first_phase_weights[27] = 0.5

        #for last phase weights

        self.winner_last_phase_weights = np.ones(TOTAL_WEIGHTS)
        for i in range(10):
            self.winner_last_phase_weights[i] = given_weights[0+NUM_OF_WEIGHTS]

        for i in range(10, 21):
            self.winner_last_phase_weights[i] = given_weights[1+NUM_OF_WEIGHTS]

        for i in range(22, 23):
            self.winner_last_phase_weights[i] = given_weights[2+NUM_OF_WEIGHTS]


        for i in range(28, 32):
            self.winner_last_phase_weights[i] = given_weights[3+NUM_OF_WEIGHTS]

            self.winner_last_phase_weights[32] = given_weights[4+NUM_OF_WEIGHTS]

        self.winner_last_phase_weights[33] = given_weights[5+NUM_OF_WEIGHTS]

        self.winner_first_phase_weights[34] = given_weights[6 +NUM_OF_WEIGHTS]

        self.winner_last_phase_weights[27] = 0.5



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


    def winning_heuristic(self, state: CatanState):
        # as discussed with Shaul, this isn't zero-sum heuristic, but a max-gain approach where only own player's
        # value is is taken in account
        self.scores_by_player = state.get_scores_by_player_indexed()
        my_score = self.scores_by_player[self.get_id()]
        if my_score >= 10:
            return inf
        max_score = max(self.scores_by_player)
        if max_score >= 10:
            return -inf
        if state.is_initialisation_phase():
            return self.heuristic_initialisation_phase(state)
        if self.in_first_phase(state):
            return self.heuristic_first_phase(state, self.winner_first_phase_weights)

        return self.heuristic_final_phase(state, self.winner_last_phase_weights)


    def in_first_phase(self, state=None):
        my_victory_points = self.scores_by_player[self.get_id()]
        return my_victory_points <= 7


    def heuristic_initialisation_phase(self, state, weights=np.ones(10)):

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
        if ore_expectation >= state.probabilities_by_dice_values[decent+1] and grain_expectation >= state.probabilities_by_dice_values[decent]:
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


    def heuristic_first_phase(self, state, weights):
        """
        prefer higher expected resource yield, rather than VP.
        also reward having places to build settlements.
        :param state: the current state of the game.
        :param weights: a np array of weights to each value that the heuristic takes into account.
        :return: returns a score for this state.
        """

        values = np.zeros(TOTAL_WEIGHTS)
        board = state.board
        scores_by_players = state.get_scores_by_player()
        currentResouces = self.resources

        # how many cards of each resource we have right now
        values[0] = currentResouces[Resource.Brick]
        values[1] = currentResouces[Resource.Lumber]
        values[2] = currentResouces[Resource.Wool]
        values[3] = currentResouces[Resource.Grain]
        values[4] = currentResouces[Resource.Ore]

        # what is our trading ratio for each resource
        brick_trade_ratio = Winner.calc_player_trade_ratio(self, state, Resource.Brick)
        lumber_trade_ratio = Winner.calc_player_trade_ratio(self, state, Resource.Lumber)
        wool_trade_ratio = Winner.calc_player_trade_ratio(self, state, Resource.Wool)
        grain_trade_ratio = Winner.calc_player_trade_ratio(self, state, Resource.Grain)
        ore_trade_ratio = Winner.calc_player_trade_ratio(self, state, Resource.Ore)

        # current resources * trade ratios
        values[5] = currentResouces[Resource.Brick] * (1 / brick_trade_ratio)
        values[6] = currentResouces[Resource.Lumber] * (1 / lumber_trade_ratio)
        values[7] = currentResouces[Resource.Wool] * (1 / wool_trade_ratio)
        values[8] = currentResouces[Resource.Grain] * (1 / grain_trade_ratio)
        values[9] = currentResouces[Resource.Ore] * (1 / ore_trade_ratio)

        resource_expectation = Winner.get_resource_expectation(self, state)

        # resource expectation
        values[10] = resource_expectation[Resource.Brick]
        values[11] = resource_expectation[Resource.Lumber]
        values[12] = resource_expectation[Resource.Wool]
        values[13] = resource_expectation[Resource.Grain]
        values[14] = resource_expectation[Resource.Ore]

        # resource expectations * trade ratios
        values[15] = resource_expectation[Resource.Brick] * (1 / brick_trade_ratio)
        values[16] = resource_expectation[Resource.Lumber] * (1 / lumber_trade_ratio)
        values[17] = resource_expectation[Resource.Wool] * (1 / wool_trade_ratio)
        values[18] = resource_expectation[Resource.Grain] * (1 / grain_trade_ratio)
        values[19] = resource_expectation[Resource.Ore] * (1 / ore_trade_ratio)

        # total resource expectation
        values[20] = np.sum(values[i] for i in range(10, 15))

        # the number of unexposed development cards, except for VP dev cards. (num dev cards)
        values[21] = sum(self.unexposed_development_cards.values()) + sum(self.exposed_development_cards.values()) - self.unexposed_development_cards[DevelopmentCard.VictoryPoint]

        # average and max difference between player's VP, and other's VP. should be with negative weights.
        values[22] = Winner.get_avg_vp_difference(scores_by_players, self)  # Avg VP diff
        values[23] = Winner.get_vp_diff(scores_by_players, self)  # Max VP diff

        values[24] = 1 if self.can_settle_settlement() else 0
        values[25] = 1 if self.can_settle_city() else 0
        values[26] = 1 if self.has_resources_for_development_card() else 0

        values[27] = len(board.get_settleable_locations_by_player(self))  # number of places we could build a settlement

        # estimate how many turns it would take to get the resources for a road, settlement, city or dev card.
        values[28] = self.get_turns_till_piece(currentResouces, resource_expectation, ResourceAmounts.road)
        values[29] = self.get_turns_till_piece(currentResouces,
                                               resource_expectation,
                                               ResourceAmounts.settlement)
        values[30] = self.get_turns_till_piece(currentResouces,
                                               resource_expectation,
                                               ResourceAmounts.city)
        values[31] = self.get_turns_till_piece(currentResouces,
                                               resource_expectation,
                                               ResourceAmounts.development_card)

        # our VP
        values[32] = scores_by_players[self]

        # the other heuristic
        values[33] = self.weighted_probabilities_heuristic(state)

        #difference between number of exposed knights.
        values[34] = self.get_exposed_knights_diff(self, state)


        values_debug = np.multiply(values,weights)
        debug = 0

        return np.dot(values, weights)


    def heuristic_final_phase(self, state, weights):

        scores_by_players = state.get_scores_by_player()
        permanent_score = self.get_victory_point_development_cards_count() + state.board.get_colonies_score(self)

        score = scores_by_players[self]

        #return self.heuristic_first_phase(state, weights)
        return score + permanent_score


    @staticmethod
    def calc_player_trade_ratio(player, state, source_resource: Resource):
        """
        return 2, 3 or 4 based on the current players harbors status
        :param source_resource: the resource the player will give
        :return: 2, 3 or 4 - the number of resource units the player will give for a single card
        """
        if state.board.is_player_on_harbor(player, Harbor(source_resource.value)):
            return 2
        if state.board.is_player_on_harbor(player, Harbor.HarborGeneric):
            return 3
        return 4


    @staticmethod
    def get_resource_expectation(player, state):
        # TODO: check that this function works properly
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


    @staticmethod
    def get_turns_till_piece(currentResources, resourceExpectation, requiredResourcesForPiece):
        """
        returns the estimation of the number of turns it would take to get all the resources to build the specified piece or dev card.
        :param resourceExpectation: a dictionary with the current resource expectation of the player (Resource -> number)
        :param currentResources: a dictionary with the resources the player has in hand at the moment.
        :param requiredResourcesForPiece: a dictionary with the required amounts for the desired piece.
        :return:
        """

        num_turns = 0

        for r in Resource:
            needed_amount = requiredResourcesForPiece[r] - currentResources[r]
            if needed_amount > 0:  # if we have the resource (in the right amount) - we are good. nothing to do
                if resourceExpectation[r] == 0:
                    num_turns += 4 / max(resourceExpectation.values())  # TODO: change to calc trade ratio for that resource
                    continue
                num_turns = max(num_turns, ceil(needed_amount / resourceExpectation[r]))

        return num_turns


    @staticmethod
    def get_avg_vp_difference(score_by_players, player):
        """
        :return: the average difference between the player's vp, and other players vp.
        """
        # vp__diff_sum = 0
        # for other in score_by_players.keys()
        return sum((score_by_players[player] - score_by_players[other]) for other in score_by_players.keys() if other != player) / (len(score_by_players) - 1)


    @staticmethod
    def get_vp_diff(score_by_player, player):
        """
        :return: the maximal difference between the player's vp, and other players vp.
        """
        max_other = max(score_by_player[other] for other in score_by_player.keys() if other != player)
        return max_other - score_by_player[player]

        #return max((score_by_player[player] - score_by_player[other]) for other in score_by_player.keys() if other != player)


    @staticmethod
    def get_exposed_knights_diff(player, state):
        max_other_knights = max(other.get_exposed_knights_count() for other in state.players if other != player)
        return max_other_knights - player.get_exposed_knights_count()


    def filter_moves(self, seed, branching_factor=3459):
        a = create_monte_carlo_filter(seed, branching_factor)
        b = self.filter_out_useless_trades()
        c = self.filter_out_robber_placements_on_self()


        def spaghetti_filter(all_moves, state):
            return a(b(c(all_moves, state), state), state)


        return spaghetti_filter


    def filter_out_useless_trades(self):

        def is_good_move(move, state):
            num_roads_before_move = self.amount_of_roads_can_afford()
            num_settlements_before_move = self.amount_of_settlements_can_afford()
            num_cities_before_move = self.amount_of_cities_can_afford()
            num_development_cards_before_move = self.amount_of_development_card_can_afford()

            for exchange in move.resources_exchanges:
                self.trade_resources(exchange.source_resource, exchange.target_resource, exchange.count,
                                     state._calc_curr_player_trade_ratio(exchange.source_resource))

            num_roads_after_move = self.amount_of_roads_can_afford()
            num_settlements_after_move = self.amount_of_settlements_can_afford()
            num_cities_after_move = self.amount_of_cities_can_afford()
            num_development_cards_after_move = self.amount_of_development_card_can_afford()

            for exchange in move.resources_exchanges:
                self.un_trade_resources(exchange.source_resource, exchange.target_resource, exchange.count,
                                        state._calc_curr_player_trade_ratio(exchange.source_resource))

            return num_roads_after_move > num_roads_before_move or \
                   num_settlements_after_move > num_settlements_before_move or \
                   num_cities_after_move > num_cities_before_move or \
                   num_development_cards_after_move > num_development_cards_before_move


        def useless_trades_filter(all_moves, state):
            good_moves = [move for move in all_moves if is_good_move(move, state)]
            if not good_moves:
                return all_moves
            return good_moves


        return useless_trades_filter


    def filter_out_robber_placements_on_self(self):

        def is_good_move(state, move) -> bool:
            if move.robber_placement_land == state.board.get_robber_land():
                return True
            for location in move.robber_placement_land.locations:
                if state.board.is_colonised_by(state.get_current_player(), location):
                    return False
            return True


        def bad_robber_placement_filter(all_moves, state):
            assert state is not None
            good_moves = [move for move in all_moves if is_good_move(state, move)]
            if not good_moves:
                return all_moves
            return good_moves


        return bad_robber_placement_filter

    def amount_of_development_card_can_afford(self):
        return min(self.resources[Resource.Ore],
                   self.resources[Resource.Wool],
                   self.resources[Resource.Grain])


    def weighted_probabilities_heuristic(self, state: CatanState):
        # TODO: fix a little
        if self._players_and_factors is None:
            self._players_and_factors = [(self, len(state.players) - 1)] + [(p, -1) for p in state.players if p is not self]

        score = 0
        # noinspection PyTypeChecker
        for player, factor in self._players_and_factors:
            for location in state.board.get_locations_colonised_by_player(player):
                weight = self.expectimax_weights[state.board.get_colony_type_at_location(location)]
                for dice_value in state.board.get_surrounding_dice_values(location):
                    score += state.probabilities_by_dice_values[dice_value] * weight * factor

            for road in state.board.get_roads_paved_by_player(player):
                weight = self.expectimax_weights[Road.Paved]
                for dice_value in state.board.get_adjacent_to_path_dice_values(road):
                    score += state.probabilities_by_dice_values[dice_value] * weight * factor

            for development_card in {DevelopmentCard.VictoryPoint, DevelopmentCard.Knight}:
                weight = self.expectimax_weights[development_card]
                score += self.get_unexposed_development_cards()[development_card] * weight * factor
        return score

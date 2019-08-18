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


class Winner(ExpectimaxBaselinePlayer):

    weights = np.ones(50)


    def __init__(self, player_id, seed=None, timeout_seconds=5):
        super().__init__(id=player_id, seed=seed, timeout_seconds=timeout_seconds, heuristic=self.tomeristic, filter_moves=self.filter_moves(seed), filter_random_moves=create_monte_carlo_filter(seed, 10))
        self.initialize_weights()


    def initialize_weights(self):
        # heavy weights for resource expectation
        for i in range(10, 20):
            self.weights[i] = 100

        # lights weights for turns to get resources for specific pieces.
        for i in range(28,32):
            self.weights[i] = 0.1

        self.expectimax_weights = {Colony.City: 2, Colony.Settlement: 1, Road.Paved: 0.4,
                                   DevelopmentCard.VictoryPoint: 1, DevelopmentCard.Knight: 2.0 / 3.0}


    def choose_resources_to_drop(self) -> Dict[Resource, int]:
        if sum(self.resources.values()) < 8:
            return {}
        resources_count = sum(self.resources.values())
        resources_to_drop_count = ceil(resources_count / 2)
        if self.can_settle_city() and resources_count >= sum(ResourceAmounts.city.values()) * 2:
            self.remove_resources_and_piece_for_city()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_and_piece_for_city()

        elif self.can_settle_settlement() and resources_count >= sum(ResourceAmounts.settlement.values()) * 2:
            self.remove_resources_and_piece_for_settlement()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_and_piece_for_settlement()

        elif (self.has_resources_for_development_card() and
              resources_count >= sum(ResourceAmounts.development_card.values()) * 2):
            self.remove_resources_for_development_card()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_for_development_card()

        elif self.can_pave_road() and resources_count >= sum(ResourceAmounts.road.values()) * 2:
            self.remove_resources_and_piece_for_road()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_and_piece_for_road()

        else:
            if sum(self.resources.values()) < 8:
                return {}
            resources = [resource for resource, resource_count in self.resources.items() for _ in range(resource_count)]
            drop_count = ceil(len(resources) / 2)
            return Counter(self._random_choice(resources, drop_count, replace=False))

        resources_to_drop = [resource for resource, count in resources_to_drop.items() for _ in range(count)]
        return Counter(self._random_choice(resources_to_drop, resources_to_drop_count, replace=False))


    def drop_resources_in_first_phase(self, state):
        return None


    def drop_resources_in_final_phase(self, state):
        return None


    def tomeristic(self, state: CatanState):
        # as discussed with Shaul, this isn't zero-sum heuristic, but a max-gain approach where only own player's
        # value is is taken in account
        if state.is_initialisation_phase():
            return self.heuristic_initialisation_phase(state)
        if self.in_first_phase(state):
            return self.heuristic_first_phase(state, self.weights)
        return self.heuristic_final_phase(state)


    def in_first_phase(self, state):
        my_victory_points = int(state.get_scores_by_player()[self])
        return my_victory_points <= 7


    def heuristic_initialisation_phase(self, state):
        return self.weighted_probabilities_heuristic(state)


    def heuristic_first_phase(self, state, weights=np.ones(50)):
        """
        prefer higher expected resource yield, rather than VP.
        also reward having places to build settlements.
        :param state: the current state of the game.
        :param weights: a np array of weights to each value that the heuristic takes into account.
        :return: returns a score for this state.
        """

        values = np.zeros(50)
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
        values[21] = len(self.unexposed_development_cards) - self.unexposed_development_cards[DevelopmentCard.VictoryPoint]

        # average and max difference between player's VP, and other's VP. should be with negative weights.
        values[22] = Winner.get_avg_vp_difference(scores_by_players, self)  # Avg VP diff
        values[23] = Winner.get_max_vp_difference(scores_by_players, self)  # Max VP diff

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

        return np.dot(values, weights)


    def heuristic_final_phase(self, state):
        scores_by_players = state.get_scores_by_player()
        can_build_dev_card = 1 if self.has_resources_for_development_card() else 0

        resource_expectation = Winner.get_resource_expectation(self, state)

        return 2 * scores_by_players[self] + 4 * can_build_dev_card + sum([resource_expectation[Resource.Brick],
                                                                           resource_expectation[Resource.Lumber],
                                                                           resource_expectation[Resource.Wool],
                                                                           resource_expectation[Resource.Grain],
                                                                           resource_expectation[Resource.Ore]])


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
                    num_turns += 4 / max(resourceExpectation.values()) #TODO: change to calc trade ratio for that resource
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
    def get_max_vp_difference(score_by_player, player):
        """
        :return: the maximal difference between the player's vp, and other players vp.
        """
        return max((score_by_player[player] - score_by_player[other]) for other in score_by_player.keys() if other != player)


    def filter_moves(self, seed, branching_factor=3459):
        a = create_monte_carlo_filter(seed, branching_factor)
        b = self.filter_out_useless_trades()
        c = create_bad_robber_placement_filter(self)


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


    def amount_of_development_card_can_afford(self):
        return min(self.resources[Resource.Ore],
                   self.resources[Resource.Wool],
                   self.resources[Resource.Grain])


    def weighted_probabilities_heuristic(self, s: CatanState):
        # TODO: fix a little
        if self._players_and_factors is None:
            self._players_and_factors = [(self, len(s.players) - 1)] + [(p, -1) for p in s.players if p is not self]

        score = 0
        # noinspection PyTypeChecker
        for player, factor in self._players_and_factors:
            for location in s.board.get_locations_colonised_by_player(player):
                weight = self.expectimax_weights[s.board.get_colony_type_at_location(location)]
                for dice_value in s.board.get_surrounding_dice_values(location):
                    score += s.probabilities_by_dice_values[dice_value] * weight * factor

            for road in s.board.get_roads_paved_by_player(player):
                weight = self.expectimax_weights[Road.Paved]
                for dice_value in s.board.get_adjacent_to_path_dice_values(road):
                    score += s.probabilities_by_dice_values[dice_value] * weight * factor

            for development_card in {DevelopmentCard.VictoryPoint, DevelopmentCard.Knight}:
                weight = self.expectimax_weights[development_card]
                score += self.get_unexposed_development_cards()[development_card] * weight * factor
        return score

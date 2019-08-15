from game.catan_state import CatanState
from players.expectimax_baseline_player import ExpectimaxBaselinePlayer
from players.expectimax_weighted_probabilities_with_filter_player import *
from players.abstract_player import *
from game.resource import *
from game.board import *
from game.development_cards import *
import math


MAX_ITERATIONS = 10


class ExpecTomer(ExpectimaxBaselinePlayer):


    def __init__(self, player_id, timeout_seconds=5):
        super().__init__(id=player_id, timeout_seconds=timeout_seconds, heuristic=self.tomeristic, filter_moves=self.filmer_toves)


    def choose_resources_to_drop(self) -> Dict[Resource, int]:
        if self.tomer_in_first_phase(state):
            return self.tomer_drops_mic_in_first_phase(state)
        return self.tomer_drops_mic_in_final_phase(state)


    def tomer_drops_mic_in_first_phase(self, state):
        return None


    def tomer_drops_mic_in_final_phase(self, state):
        return None


    def filmer_toves(self, state):
        pass


    def tomeristic(self, state: CatanState):
        # as discussed with Shaul, this isn't zero-sum heuristic, but a max-gain approach where only own player's
        # value is is taken in account
        if state.is_initialisation_phase():
            return self.tomeristic_initialisation_phase(state)
        if self.tomer_in_first_phase(state):
            return self.tomeristic_first_phase(state)
        return self.tomeristic_final_phase(state)


    def tomer_in_first_phase(self, state):
        my_victory_points = int(state.get_scores_by_player()[self])
        return my_victory_points <= 6


    def tomeristic_initialisation_phase(self, state):
        pass


    def tomeristic_first_phase(self, state):
        """
        prefer higher expected resource yield, rather than VP.
        also reward having places to build settlements.
        :param state: the current state of the game.
        :param player: our player.
        :return: returns a score for this state.
        """
        board = state.board
        scores_by_players = state.get_scores_by_player()

        # how many cards of each resource we have right now
        brick_count = self.get_resource_count(Resource.Brick)
        lumber_count = self.get_resource_count(Resource.Lumber)
        wool_count = self.get_resource_count(Resource.Wool)
        grain_count = self.get_resource_count(Resource.Grain)
        ore_count = self.get_resource_count(Resource.Ore)

        # what is our trading ratio for each resource
        brick_trade_ratio = ExpecTomer.calc_player_trade_ratio(self, state, Resource.Brick)
        lumber_trade_ratio = ExpecTomer.calc_player_trade_ratio(self, state, Resource.Lumber)
        wool_trade_ratio = ExpecTomer.calc_player_trade_ratio(self, state, Resource.Wool)
        grain_trade_ratio = ExpecTomer.calc_player_trade_ratio(self, state, Resource.Grain)
        ore_trade_ratio = ExpecTomer.calc_player_trade_ratio(self, state, Resource.Ore)

        # the number of unexposed development cards, except for VP dev cards.
        num_dev_cards = sum(self.unexposed_development_cards) - self.unexposed_development_cards[DevelopmentCard.VictoryPoint]

        avg_vp_diff = ExpecTomer.get_avg_vp_difference(scores_by_players, self)
        max_vp_diff = ExpecTomer.get_max_vp_difference(scores_by_players, self)

        can_build_settlement = 1 if self.can_settle_settlement() else 0
        can_build_city = 1 if self.can_settle_city() else 0
        can_build_dev_card = 1 if self.has_resources_for_development_card() else 0

        # resource expectation!!
        num_places_to_build = len(board.get_settleable_locations_by_player())


    def tomeristic_final_phase(self, state):
        pass


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
    def get_resource_expectation():
        """
        calculates the expected resource yield per one turn per player.
        :return:
        """


    @staticmethod
    def get_avg_vp_difference(score_by_players, player):
        """
        :return: the average difference between the player's vp, and other players vp.
        """
        return sum(score_by_players[player] - score_by_players[other] for other in score_by_players.keys if other != player) / len(score_by_players)


    @staticmethod
    def get_max_vp_difference(score_by_player, player):
        """
        :return: the maximal difference between the player's vp, and other players vp.
        """
        return max(score_by_player[player] - score_by_player[other] for other in score_by_player.keys if other != player)

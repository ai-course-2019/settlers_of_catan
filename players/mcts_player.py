from math import ceil

from game.catan_state import CatanState
from game.resource import Resource, ResourceAmounts

from players.abstract_player import AbstractPlayer
from algorithms.mcts import MCTS, MCTSNode
from collections import Counter

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
            return self._random_choice(next_moves)  # TODO: change to the better heuristic
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


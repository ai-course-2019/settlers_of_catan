from players.expectimax_weighted_probabilities_player import ExpectimaxWeightedProbabilitiesPlayer
from players.filters import create_monte_carlo_filter


class MonteCarloPlayer(ExpectimaxWeightedProbabilitiesPlayer):
    def __init__(self, id, seed=None, timeout_seconds=5, branching_factor=3459):
        super().__init__(id,
                         seed=seed,
                         timeout_seconds=timeout_seconds,
                         filter_moves=create_monte_carlo_filter(seed, branching_factor))

import os
import time

from game.catan_state import CatanState
from players.expectimax_baseline_player import ExpectimaxBaselinePlayer
from players.random_player import RandomPlayer
from train_and_test.logger import logger, fileLogger
from players.mcts_player import MCTSPlayer


def scores_changed(state, previous_scores, scores):
    for player in state.players:
        if previous_scores[player.get_id()] != scores[player.get_id()]:
            return True
    return False


def execute_game(plot_map=True):
    seed = None
    timeout_seconds = 5
    p0 = MCTSPlayer(0, iterations=100)
    p1 = RandomPlayer(1)
    p2 = RandomPlayer(2)
    p3 = RandomPlayer(3)
    players = [p0, p1, p2, p3]

    state = CatanState(players, seed)

    turn_count = 0
    score_by_player = state.get_scores_by_player_indexed()
    # if plot_map:
    #     state.board.plot_map('turn_{}_scores_{}.png'
    #                          .format(turn_count, ''.join('{}_'.format(v) for v in score_by_player.values())))

    while not state.is_final():
        # noinspection PyProtectedMember
        logger.info('----------------------p{}\'s turn----------------------'.format(state._current_player_index))

        turn_count += 1
        robber_placement = state.board.get_robber_land()

        move = state.get_current_player().choose_move(state)
        assert not scores_changed(state, score_by_player, state.get_scores_by_player_indexed())
        state.make_move(move)
        state.make_random_move()

        score_by_player = state.get_scores_by_player_indexed()

        move_data = {k: v for k, v in move.__dict__.items() if (v and k != 'resources_updates') and not
                     (k == 'robber_placement_land' and v == robber_placement) and not
                     (isinstance(v, dict) and sum(v.values()) == 0)}
        logger.info('| {}| turn: {:3} | move:{} |'.format(''.join('{} '.format(v) for v in score_by_player),
                                                          turn_count, move_data))
        # if plot_map:
        #     image_name = 'turn_{}_scores_{}.png'.format(
        #         turn_count, ''.join('{}_'.format(v) for v in score_by_player))
        #     state.board.plot_map(image_name, state.current_dice_number)

    players_scores_by_names = {(k, v.__class__, v.expectimax_alpha_beta.evaluate_heuristic_value.__name__ if (
        isinstance(v, ExpectimaxBaselinePlayer)) else None): score_by_player[v.get_id()]
                               for k, v in locals().items() if v in players
                               }
    fileLogger.info('\n' + '\n'.join(' {:80} : {} '.format(str(name), score)
                                     for name, score in players_scores_by_names.items()) +
                    '\n turns it took: {}\n'.format(turn_count) + ('-' * 156))

    p0_type = type(p0).__name__
    p_others_type = type(p1).__name__



def run_single_game_and_plot_map():
    execute_game(plot_map=False)


if __name__ == '__main__':
    run_single_game_and_plot_map()

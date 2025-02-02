import os
import time

from game.catan_state import CatanState
from players.expectimax_baseline_player import ExpectimaxBaselinePlayer
from players.monte_carlo_with_filter_player import MonteCarloWithFilterPlayer
from players.expectimax_weighted_probabilities_with_filter_player import *
from players.expectimax_drop_resource_cards_player import *
from players.expectimax_weighted_probabilities_player import *
from players.monte_carlo_player import *
from players.monte_carlo_with_filter_player import *
from players.random_player import RandomPlayer
from train_and_test.logger import logger, fileLogger
from players.winner import Winner


A, B, C, D, E, F, G = [], [], [], [], [], [], []
excel_file_name = 'NAME_NOT_SET_{}.xlsx'.format(time.time())

TIME_OUT = 5


def excel_data_grabber(a, b, c, d, e, f, g):
    global A, B, C, D, E, F, G
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)
    E.append(e)
    F.append(f)
    G.append(g)


def scores_changed(state, previous_scores, scores):
    for player in state.players:
        if previous_scores[player.get_id()] != scores[player.get_id()]:
            return True
    return False


def clean_previous_images():
    for file_name in os.listdir(path='.'):
        if file_name.split(sep='_')[0] == 'turn':
            os.remove(file_name)


def execute_game(plot_map=True):
    seed = None
    timeout_seconds = TIME_OUT
    p0 = Winner(id=0, seed=seed, timeout_seconds=timeout_seconds)
    p1 = ExpectimaxWeightedProbabilitiesWithFilterPlayer(id=1, seed=seed, timeout_seconds=timeout_seconds)
    p2 = RandomPlayer(id=2)
    p3 = RandomPlayer(id=3)
    players = [p0, p1, p2, p3]

    state = CatanState(players, seed)

    turn_count = 0
    score_by_player = state.get_scores_by_player_indexed()

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
        if plot_map:
            image_name = 'turn_{}_scores_{}.png'.format(
                turn_count, ''.join('{}_'.format(v) for v in score_by_player))
            state.board.plot_map(image_name, state.current_dice_number)

    players_scores_by_names = {(k, v.__class__, v.expectimax_alpha_beta.evaluate_heuristic_value.__name__ if (
        isinstance(v, ExpectimaxBaselinePlayer)) else None): score_by_player[v.get_id()]
                               for k, v in locals().items() if v in players
                               }
    names = list(players_scores_by_names.keys())
    names.sort()

    fileLogger.info('\n' + '\n'.join(' {:80} : {} '.format(str(name), players_scores_by_names[name])
                                     for name in names) +
                    '\n turns it took: {}\n'.format(turn_count) + ('-' * 156))

    p0_type = type(p0).__name__
    p_others_type = type(p1).__name__
    global excel_file_name
    excel_file_name = '{}_vs_{}_timeout_{}_seed_{}.xlsx'.format(p0_type, p_others_type, timeout_seconds, seed,
                                                                int(time.time()))
    excel_data_grabber(score_by_player[0], score_by_player[1], score_by_player[2], score_by_player[3], turn_count,
                       p0_type, p_others_type)

    excel_output = ""
    for i in range(len(players)):
        player_output = str(players[i]) + "@" + str(score_by_player[i])
        excel_output += player_output + "\n"
    # fileLogger.info("|\n#" + str(turn_count) + "\n" + excel_output)

    if score_by_player[0] >= 10:
        return 1
    else:
        return 0


def flush_to_excel():
    global A, B, C, D, E, F, G, excel_file_name
    import pandas as pd

    df = pd.DataFrame({"p0_score": A, "p1_score": B, "p2_score": C,
                       "p3_score": D, "turn_count": E, "p0_type": F, "p1_2_3_type": G})
    writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='results')


def run_10_games_parallel():
    global A, B, C, D, E, F, G
    import multiprocessing

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    pool = multiprocessing.Pool(processes=cpus)
    pool.map(execute_game, [False] * 10)

    # for _ in range(10):
    #     clean_previous_images()
    #     execute_game(None)
    flush_to_excel()


def run_single_game_and_plot_map():
    execute_game(plot_map=False)


def run_n_games(n):
    count_wins = 0
    for i in range(n):
        count_wins += execute_game(plot_map=False)
    # print("\n==============================================================\n"
    #       + "Number of wins (out of " + str(n) + "): \n"
    #       + str(count_wins)
    #       + "\n==============================================================\n")


if __name__ == '__main__':
    run_n_games(1)

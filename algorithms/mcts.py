import copy
import numpy as np

from game.catan_state import CatanState
from game.catan_moves import CatanMove


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, root: MCTSNode, moves, exploration_weight=1.4):
        assert not root.is_terminal()
        self.root = root
        for move in moves:
            my_state = copy.deepcopy(self.root.state)
            my_state.make_move(move)
            my_state.make_random_move()
            self.root.children.append(MCTSNode(my_state, move, self.root))
        self.player = self.root.state.get_current_player_index()
        self.children = {self.root: self.root.children}  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self):
        "Choose the best successor of node. (Choose a move in the game)"
        return self.root.best_child(0.)

    def do_rollout(self):
        "Make the tree one layer better. (Train for one iteration.)"
        chosen_child = self._select()
        result = 1 if (self.player == self._simulate(chosen_child)) else 0
        chosen_child.backpropagate(result)

    def do_n_rollouts(self, n):
        for _ in range(n):
            self.do_rollout()

    def _select(self):
        "Find an unexplored descendant"
        node = self.root
        while True:
            if node.get_n() == 0 or node.is_terminal():
                return node
            keys = self.children.keys()
            if node not in keys:
                self._expand(node)
            unexplored = self.children[node] - keys
            if unexplored:
                return unexplored.pop()
            node = node.best_child(self.exploration_weight)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the result of a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                return node.result()
            node = node.find_random_child()


class MCTSNode:
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    """
    def __init__(self, state: CatanState, move: CatanMove=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = list()
        self._q = 0
        self._n = 0

    def find_children(self):
        "All possible successors of this board state"
        all_moves = self.state.get_next_moves()
        for move in all_moves:
            my_state = copy.deepcopy(self.state)
            my_state.make_move(move)
            my_state.make_random_move()
            self.children.append(MCTSNode(my_state, move, self))
        return self.children

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        move = self.state.get_random_move()
        curr_state = copy.deepcopy(self.state)
        curr_state.make_move(move)
        curr_state.make_random_move()
        random_child = MCTSNode(curr_state, move, self)
        return random_child

    def best_child(self, c_param=1.4):
        children_weights = []
        for child in self.children:
            if not child.get_n() == 0:
                children_weights.append(
                    (child.get_q() / child.get_n()) + c_param *
                    np.sqrt((2 * np.log(self._n) / child.get_n())))
            else:
                children_weights.append(-np.math.inf)
        return self.children[children_weights.index(max(children_weights))]

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.state.is_final()

    def result(self):
        "Assumes `self` is terminal node. returns winner player's index"
        scores_by_players = self.state.get_scores_by_player_indexed()
        return scores_by_players.index(max(scores_by_players))

    def backpropagate(self, result):
        self._n += 1
        self._q += result
        if self.parent:
            self.parent.backpropagate(result)

    def get_q(self):
        return self._q

    def get_n(self):
        return self._n

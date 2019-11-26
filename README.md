# Final Project in AI - Settlers of Catan

# Authors
    - Ofer Ravid
    - Ethan Tempel
    - Tomer Micahel
    - Erez Levy


# Files
    - algorithms:
        - mcts.py - this file defines the algorithm for the Monte Carlo Tree Search, using 2 classes:
                    MCTS and MCTSNode.
        - the rest of the files are as is from the source code from github
    - game:
        - catan_state.py - most of the file is as is from the source code except a couple of getters especially
                           the function get_random_move and several protected methods that this function uses. 
						   the function get _random_move is used by MCTS for the simulation part.
        - the rest of the files are as is from the source code from github
    - players:
        - mcts_player.py - defining the mcts player using the MCTS algorithm for choosing moves.
        - winner.py - defining our expectimax with alpha beta pruning player, using heuristics.
        - the rest of the files are as is from the source code from github
    - train_and_test:
        - main.py - the file to run games, in the function execute_game you choose your players, from the players in
                    the players package, and in the main function you decide how many games to run.


# General Information About The Code
    The game is defined as an object of CatanState (game.catan_state.py), that contains 3-4 player objects,
    depends on the players given by the main function, a Board object (game.board.py) which contains enumerators:
    Pieces, DevelopmentCards and Resource, the moves used to update the state of the game are given through the class
    CatanMove (game.catan_move.py).
	
	At the moment running the game is only possible on machines using Linux OS, because some of the python packages
	used in the code, doesn't work properly on Windows OS.
	
	Another problem is that the code we found for the game uses networkx.graph to create the board, and to draw it
	we need to use the package pygraphviz, however we couldn't install it on any OS, so at the moment it is impossible
	to display the game board, but we plan in the future to rewrite the code so it will be possible.

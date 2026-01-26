import random

class TicTacToeEnv:
    EMPTY = 0
    X = 1
    O = -1

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [self.EMPTY] * 9
        self.done = False
        return tuple(self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == self.EMPTY]

    def step(self, action, player):
        if self.board[action] != self.EMPTY:
            return tuple(self.board), -10, True

        self.board[action] = player

        if self.check_winner(player):
            return tuple(self.board), 1, True

        if not self.available_actions():
            return tuple(self.board), 0, True

        return tuple(self.board), 0, False

    def check_winner(self, p):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        return any(self.board[a] == self.board[b] == self.board[c] == p for a,b,c in wins)
    
    def get_winner(self):
        if self.check_winner(self.X):
            return "X"
        elif self.check_winner(self.O):
            return "O"
        elif not self.available_actions():
            return "Draw"
        else:
            return None

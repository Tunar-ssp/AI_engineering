
PLOT_PATH = "models/training1/training_plot.png"

import pygame
import numpy as np
from neural_network import model
from game.run import TicTacToeEnv

MODEL_PATH = "models/training2/model.npz"

CELL_SIZE = 100
GRID_SIZE = 3
SCREEN_SIZE = CELL_SIZE * GRID_SIZE
LINE_WIDTH = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("TicTacToe AI vs Human")
font = pygame.font.SysFont(None, 72)

def draw_grid(board):
    screen.fill(WHITE)
    for x in range(1, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (x * CELL_SIZE, 0), (x * CELL_SIZE, SCREEN_SIZE), LINE_WIDTH)
    for y in range(1, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, y * CELL_SIZE), (SCREEN_SIZE, y * CELL_SIZE), LINE_WIDTH)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
            if board[y, x] == 1:
                text = font.render("X", True, RED)
                text_rect = text.get_rect(center=center)
                screen.blit(text, text_rect)
            elif board[y, x] == -1:
                text = font.render("O", True, BLUE)
                text_rect = text.get_rect(center=center)
                screen.blit(text, text_rect)

net = model()
net.load_model(MODEL_PATH)

def ai_move(state, available_actions):
    q_vals = net.forward_propagation(np.array(state))[0]
    q_vals_masked = np.full_like(q_vals, -np.inf)
    q_vals_masked[available_actions] = q_vals[available_actions]
    return int(np.argmax(q_vals_masked))

def play_human_vs_ai():
    env = TicTacToeEnv()
    state = env.reset()
    state = np.array(state)
    done = False
    current_player = 1
    winner = None

    draw_grid(state.reshape(3, 3))
    pygame.display.flip()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN and current_player == -1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                action = row * GRID_SIZE + col
                if action in env.available_actions():
                    state, reward, done = env.step(action, current_player)
                    current_player *= -1
                    state = np.array(state)
                    draw_grid(state.reshape(3, 3))
                    pygame.display.flip()

        if current_player == 1 and not done:
            available_actions = env.available_actions()
            action = ai_move(state, available_actions)
            state, reward, done = env.step(action, current_player)
            current_player *= -1
            state = np.array(state)
            draw_grid(state.reshape(3, 3))
            pygame.display.flip()
            pygame.time.delay(500)

    winner = env.get_winner() if hasattr(env, "get_winner") else None
    print("Game over! Winner:", winner)

    text = font.render(f"Winner: {winner}", True, BLACK)
    text_rect = text.get_rect(center=(SCREEN_SIZE//2, SCREEN_SIZE//2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.delay(3000)

if __name__ == "__main__":
    play_human_vs_ai()
    pygame.quit()

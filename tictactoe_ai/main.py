import os
import numpy as np
import matplotlib.pyplot as plt
from random import random, choice
from game.run import TicTacToeEnv
from neural_network import model

def plot_details(game_length, rewards, loss, filename):
    epochs = range(1, len(game_length) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, game_length, label='Game Length')
    plt.xlabel('Epochs')
    plt.ylabel('Length')
    plt.title('Game Length over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, rewards, label='Rewards', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.title('Rewards over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, loss, label='Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    env = TicTacToeEnv()
    net = model()
    base_folder = "models"
    prefix = "training"

    epochs = 15000
    learning_rate = 0.0001
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.02

    game_lengths = []
    rewards_list = []
    losses_list = []

    for ep in range(epochs):
        state = np.array(env.reset())
        done = False
        current_player = 1
        ep_reward = 0
        ep_loss = 0
        moves = 0

        while not done:
            available_actions = env.available_actions()

            if random() < epsilon:
                action = choice(available_actions)
            else:
                q_vals = net.forward_propagation(state)[0]
                q_vals_masked = np.full_like(q_vals, -np.inf)
                q_vals_masked[available_actions] = q_vals[available_actions]
                action = np.argmax(q_vals_masked)

            next_state, reward, done = env.step(action, current_player)
            next_state = np.array(next_state)
            Q_pred = net.forward_propagation(state)[0]
            Q_target = Q_pred.copy()
            if done:
                Q_target[action] = reward
            else:
                Q_next = net.forward_propagation(next_state)[0]
                Q_target[action] = reward + gamma * np.max(Q_next)

            loss_grad = (Q_pred - Q_target).reshape(1, -1)
            net.gradient_descent(loss_grad)
            net.backward_propagation(learning_rate)

            ep_loss += np.sum((Q_pred - Q_target)**2)
            ep_reward += reward
            moves += 1

            state = next_state
            current_player *= -1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        game_lengths.append(moves)
        rewards_list.append(ep_reward)
        losses_list.append(ep_loss)

    i = 1
    while os.path.exists(os.path.join(base_folder, f"{prefix}{i}")):
        i += 1
    model_folder = os.path.join(base_folder, f"{prefix}{i}")
    os.makedirs(model_folder)

    model_file = os.path.join(model_folder, "model")
    net.save_model(model_file)
    print(f"Model saved to {model_file}")

    plot_file = os.path.join(model_folder, "training_plot.png")
    plot_details(game_lengths, rewards_list, losses_list, plot_file)
    print(f"Training plot saved to {plot_file}")

if __name__ == "__main__":
    main()

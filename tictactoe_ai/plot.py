import matplotlib.pyplot as plt

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
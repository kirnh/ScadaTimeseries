import matplotlib.pyplot as plt


def plot_graphs(arrays, array_labels, save_path):
    if len(arrays) != len(array_labels):
        print("Number of arrays should match number of labels...")
        return None
    fig = plt.figure()
    fig.set_size_inches([600, 10])
    for array in arrays:
        plt.plot(array)
    plt.legend(array_labels, loc="upper left")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


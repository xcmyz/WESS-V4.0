import matplotlib.pyplot as plt
import numpy as np


def visualize(loss_file_name):
    loss_arr = np.array(list())
    with open(loss_file_name, "r") as f_loss:
        for loss in f_loss.readlines():
            loss_arr = np.append(loss_arr, float(loss))

    x = np.array([i for i in range(np.shape(loss_arr)[0])])
    y = loss_arr

    plt.figure()
    plt.plot(x, y, color="y", lw=0.7)

    plt.xlabel("sequence number")
    plt.ylabel("total loss item")
    plt.title("total loss")
    plt.savefig("loss.jpg")


if __name__ == "__main__":
    # Test
    visualize("loss.txt")

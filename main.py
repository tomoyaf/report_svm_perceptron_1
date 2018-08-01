import matplotlib.pyplot as plt
import numpy as np

def update_alpha_i(alpha, gamma, x, y, i):
    return alpha[i] + gamma * (1.0 - np.sum([alpha[j] * y[i] * y[j] * x[i].T * x[j] for j in range(y.shape[0])]))

def update_alpha(alpha, gamma, x, y):
    return  [update_alpha_i(alpha, gamma, x, y, i) for i in range(y.shape[0])]

def train_svm(x, y, num_of_epoch=100, gamma=0.01):
    alpha = np.zeros(y.shape)
    for _ in range(num_of_epoch):
        alpha = update_alpha(alpha, gamma, x, y)
    return np.sum(np.array([alpha[i] * y[i] * x[i] for i in range(y.shape[0])]), axis=0)

def train_perceptron(x, y, num_of_epoch=100, gamma=0.001):
    w = np.zeros(x[0].T.shape)
    for _ in range(num_of_epoch):
        for i in range(y.shape[0]):
            w += gamma * y[i] * x[i]
    return w

if __name__ == "__main__":
    x = np.array([
        [-1.0,  1.0],
        [-1.0,  2.0],
        [-2.0,  1.0],
        [-2.0,  2.0],
        [ 1.0, -1.0],
        [ 1.0, -2.0],
        [ 2.0, -1.0],
        [ 2.0, -2.0],
    ])
    y = np.array([
        -1, -1, -1, -1,
         1,  1,  1,  1
    ])

    w_svm = train_svm(x, y)
    w_perceptron = train_perceptron(x, y)

    plt.figure()
    plt.suptitle("Figure 1")

    positive_x = [x[i][0] for i in range(y.shape[0]) if y[i] > 0]
    positive_y = [x[i][1] for i in range(y.shape[0]) if y[i] > 0]
    negative_x = [x[i][0] for i in range(y.shape[0]) if y[i] < 0]
    negative_y = [x[i][1] for i in range(y.shape[0]) if y[i] < 0]

    plt.subplot(211)
    plt.title("SVM w=" + str(w_svm))
    plt.plot(positive_x, positive_y, "r*")
    plt.plot(negative_x, negative_y, "b*")
    plt.plot([-2.0, 2.0], -w_svm[0] / w_svm[1] * np.array([-2.0, 2.0]), "k-")

    plt.subplot(212)
    plt.title("Perceptron w=" + str(w_perceptron))
    positive_x = [x[i][0] for i in range(y.shape[0]) if y[i] > 0]
    positive_y = [x[i][1] for i in range(y.shape[0]) if y[i] > 0]
    negative_x = [x[i][0] for i in range(y.shape[0]) if y[i] < 0]
    negative_y = [x[i][1] for i in range(y.shape[0]) if y[i] < 0]
    plt.plot(positive_x, positive_y, "r*")
    plt.plot(negative_x, negative_y, "b*")
    plt.plot([-2.0, 2.0], -w_perceptron[0] / w_perceptron[1] * np.array([-2.0, 2.0]), "k-")

    plt.show()

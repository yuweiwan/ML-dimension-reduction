import numpy as np


def txt_to_matrix(filename):
    file = open(filename)
    lines = file.readlines()
    datamat = np.zeros((800, 13))
    row = 0
    for line in lines:
        line = line.split(" ")
        line = [float(x) for x in line]
        datamat[row, :] = line[:]
        row += 1
    return datamat


def compute_mean(matrix):
    m = []
    for d in range(13):
        m.append(sum(matrix[:, d]) / 800)
    return m


def compute_Dcovariance(matrix, mean):
    D = []
    for d in range(13):
        D.append(sum((matrix[:, d] - mean[d]) ** 2) / 800)
    return D
def compute_loglikelihood(matrix, m, D):
    log = np.zeros((800, 13))
    for t in range(800):
        for d in range(13):
            log[t][d] = np.log((2 * np.math.pi*D[d])**(-1/2) * np.math.exp(-(matrix[t][d]-m[d])**2 / (2*D[d])))
    return sum(sum(log))


matrix = txt_to_matrix("mfcc13.txt")
mean = compute_mean(matrix)
Dcovariance = compute_Dcovariance(matrix, mean)
loglikelihood=compute_loglikelihood(matrix, mean, Dcovariance)

pi = np.zeros(5)
pi[0]=1

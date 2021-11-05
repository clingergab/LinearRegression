import statistics
import numpy as np
import pandas as pd
import sys
import copy
from plot_db import visualize_3d

class lr:
    b = []
    def __init__(self):
        self.b = [0, 0, 0]

    def prep(self, data):
        data.insert(0, 'bias', 1)
        data[0] = (data[0]-data[0].mean()) / data[0].std()
        data[1] = (data[1]-data[1].mean()) / data[1].std()

    def regression(self, data, alpha):
        
        for i in range(len(self.b)):
            sum = 0
            for item in data:
                sum += (self.fx(item) - item[3]) * item[i]
            self.b[i] = self.b[i] - (alpha/len(data) * sum)
            

    def fx(self, input):
        fx = self.b[0]
        for i in range(1, 3):
            fx += self.b[i] * input[i]
        return fx

    def rb(self, data):
        rb = 0
        for item in data:
            rb += (self.fx(item) - item[3]) ** 2
        rb /= (2 * len(data))
        return rb

def main():
    f = open(sys.argv[2], "w")
    csvFile = pd.read_csv(sys.argv[1], header=None)
    reg = lr()
    reg.prep(csvFile)
    data = np.array(csvFile)
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.7]
    for alpha in alphas:
        for i in range(100):
            reg.regression(data, alpha)
        f.write(str(alpha) + ' ')
        f.write(str(100) + ' ')
        f.write(str(reg.b) + "\n")
        #print("alpha: " + str(alpha) + " R: " + str(reg.rb(data)))
        visualize_3d(csvFile, reg.b, feat1=0, feat2=1, labels=2,
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=('Gradient Descent alpha: ' + str(alpha)))

    f.close()
    

if __name__ == "__main__":
    main()



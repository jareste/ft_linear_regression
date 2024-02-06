import csv
import sys
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class FtLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.theta0 = 0
        self.theta1 = 0
        self.learning_rate = learning_rate
        self.iterations = iterations

    def estimate_price(self, mileage):
        return self.theta0 + self.theta1 * mileage

    def fit(self, x, y):
        m = len(x)
        x_norm = (x - np.mean(x)) / (np.max(x) - np.min(x))  
        y_norm = (y - np.mean(y)) / (np.max(y) - np.min(y))  
        for i in range(self.iterations):
            h = self.theta0 + self.theta1 * x_norm
            loss = h - y_norm
            gradient_theta0 = np.sum(loss) / m
            gradient_theta1 = np.dot(x_norm, loss) / m
            self.theta1 -= self.learning_rate * gradient_theta1
            self.theta0 -= self.learning_rate * gradient_theta0
        self.theta0 = self.theta0 * (np.max(y) - np.min(y)) + np.mean(y) - self.theta1 * np.mean(x) * (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))
        self.theta1 = self.theta1 * (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))

    def predict(self, x):
        return [self.estimate_price(xi) for xi in x]

    def print(self):
        print('theta0:', self.theta0)
        print('theta1:', self.theta1)

    def save(self, filename):
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([self.theta0, self.theta1])

# import csv

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = {header: [] for header in headers}
        for row in reader:
            for header, value in zip(headers, row):
                data[header].append(value)
    return data



def main():
    sns.set(style='darkgrid')
    try:
        data = read_csv('data.csv')
        print (data)
        if data == {}:
            raise Exception
    except:
        print('error: data.csv not found or invalid format')
        sys.exit(1)

    data['km'] = [float(i) for i in data['km']]
    data['price'] = [float(i) for i in data['price']]

    x = data['km']
    y = data['price']

    model = FtLinearRegression()
    model.fit(x, y)
    data['predicted'] = model.predict(data['km'])

    sns.scatterplot(x='km', y='price', data=pd.DataFrame(data))
    sns.lineplot(x='km', y='predicted', color='orange', lw=4, data=pd.DataFrame(data))

    model.print()
    model.save('model.csv')

    plt.show()

if __name__ == '__main__':
    main()
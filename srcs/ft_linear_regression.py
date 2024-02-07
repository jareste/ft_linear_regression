import csv
import sys
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statistics

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
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        x_norm = [(xi - mean_x) / (max_x - min_x) for xi in x]
        y_norm = [(yi - mean_y) / (max_y - min_y) for yi in y]
        for i in range(self.iterations):
            h = [self.theta0 + self.theta1 * xi for xi in x_norm]
            loss = [hi - yi for hi, yi in zip(h, y_norm)]
            gradient_theta0 = sum(loss) / m
            gradient_theta1 = sum(xi * li for xi, li in zip(x_norm, loss)) / m
            self.theta1 -= self.learning_rate * gradient_theta1
            self.theta0 -= self.learning_rate * gradient_theta0
        self.theta0 = self.theta0 * (max_y - min_y) + mean_y - self.theta1 * mean_x * (max_y - min_y) / (max_x - min_x)
        self.theta1 = self.theta1 * (max_y - min_y) / (max_x - min_x)

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

    plt.title('Price vs Mileage')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.scatter(x, y)
    plt.plot(x, data['predicted'], color='orange', linewidth=4)

    model.print()
    model.save('model.csv')

    plt.show()

if __name__ == '__main__':
    main()
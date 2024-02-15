import csv
import sys
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statistics
import argparse

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
        print('theta0 and theta1 saved to', filename)

    def rmse(self, y_true, y_pred): # Root Mean Squared Error
        return math.sqrt(sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true))

    def mae(self, y_true, y_pred): # Mean Absolute Error
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

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
    parse = argparse.ArgumentParser(
                    prog='python3 ft_linear_regression.py',
                    description='This was my first linear regression program. It reads a csv file with two columns: \
                    km and price. It then fits a linear model to the data and prints the model parameters. \
                    Optionally, it can print the error metrics and plot the data. \
                    The model parameters are saved to model.csv. The program requires the seaborn and matplotlib libraries.',
                    epilog='Follow me on github: https://github.com/jareste.')
    parse.add_argument('-e', '--error', action='store_true', help='Print the error metrics')
    parse.add_argument('-p', '--plot', action='store_true', help='Plot the data')
    args = parse.parse_args()
    try:
        data = read_csv('data.csv')
        if data == {}:
            raise Exception
    except:
        print('error: data.csv not found or invalid format')
        sys.exit(1)

    try:
        data['km'] = [float(i) for i in data['km']]
        data['price'] = [float(i) for i in data['price']]
    except:
        print('error: invalid data format')
        sys.exit(1)
    x = data['km']
    y = data['price']

    model = FtLinearRegression()
    try:
        model.fit(x, y)
    except:
        print('Error: failed to fit the model.')
        sys.exit(1)
    data['predicted'] = model.predict(data['km'])


    model.print()
    try:
        model.save('model.csv')
    except:
        print('Error: failed to save the model.')
        sys.exit(1)

    if args.error:
        rmse = model.rmse(y, data['predicted'])
        mae = model.mae(y, data['predicted'])
        print('Root mean square error:', rmse, "euros.")
        print('Mean absolute error:', mae, "euros.")

    if args.plot:
        sns.set(style='darkgrid')
        plt.title('Price vs Mileage')
        plt.xlabel('Mileage (km)')
        plt.ylabel('Price (€)')
        plt.scatter(x, y)
        plt.plot(x, data['predicted'], color='orange', linewidth=4)
        plt.show()

    if plt:
        plt.close()

if __name__ == '__main__':
    main()
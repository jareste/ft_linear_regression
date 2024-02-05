import csv

def get_parameters(file_name):
    try:
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            theta0, theta1 = next(reader)
        try:
            theta0, theta1 = next(reader)
        except StopIteration:
            print("The file is empty. Using default values for theta0 and theta1.")
            theta0, theta1 = 0.0, 0.0
    except FileNotFoundError:
        print("The file does not exist. Using default values for theta0 and theta1.")
        theta0, theta1 = 0.0, 0.0
    return float(theta0), float(theta1)

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    theta0, theta1 = get_parameters('model_parameters.csv')
    mileage = float(input("Enter a mileage: "))
    price = estimate_price(mileage, theta0, theta1)
    print(f"The estimated price for a car with {mileage} mileage is {price}")

if __name__ == "__main__":
    main()
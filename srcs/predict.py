import csv

def get_parameters(file_name):
    try:
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            theta0, theta1 = next(reader)
    except StopIteration:
        print("The file is empty. Using default values for theta0 and theta1.")
        theta0, theta1 = 0.0, 0.0
    except FileNotFoundError:
        print("The file does not exist. Using default values for theta0 and theta1.")
        theta0, theta1 = 0.0, 0.0
    except:
        print("An error occurred while reading the file. Using default values for theta0 and theta1.")
        theta0, theta1 = 0.0, 0.0
    return float(theta0), float(theta1)

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    theta0, theta1 = get_parameters('model.csv')
    while True:
        try:
            mileage = float(input("Enter a mileage: "))
        except:
            print("Invalid mileage. please introduce a valid one.")
            continue
        if mileage < 0:
            print("The mileage must be a positive number.")
            continue
        break
    price = estimate_price(mileage, theta0, theta1)
    if price < 0:
        price = 0
    print(f"The estimated price for a car with {mileage} mileage is {price} euros.")

if __name__ == "__main__":
    main()
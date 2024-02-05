import csv
import sys
import os.path
import matplotlib.pyplot as plt
import numpy as np

class linearRegression:
    self.getData()
    self.theta0 = 0
    self.theta1 = 0

    def getData(self):
        self.values = []
        if os.path.isfile('data.csv') == False:
            print("Error: File does not exist")
            sys.exit(1)
        if os.access('data.csv', os.R_OK) == False:
            print("Error: File is not readable")
            sys.exit(1)
        with open('data.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.values.append(row)


def main():
    if len(sys.argv) < 2:
        print("Error: No file provided")
        sys.exit(1)
    
    data = linearRegression()
    data.

if __name__ == "__main__":
    main()

# #read csv file
# data = []

# with open('data.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         data.append(row)

# # print(data)

# m = 0
# b = 0

# for i in range(1, len(data)):
#     m += (int(data[i][0]) - int(data[1][0])) * (int(data[i][1]) - int(data[1][0]))
#     b += (int(data[i][0]) - int(data[1][0])) ** 2
#     print(m, b)

# x = []
# y = []

# for i in range(1, len(data)):
#     x.append(int(data[i][0]))
#     y.append(int(data[i][1]))


# # plt.plot(m, b, 'o')
# plt.plot(x, y, 'o')
# plt.show()
# # y = m*x+ b




#print graph
# # importing the required module
# import matplotlib.pyplot as plt
 
# # x axis values
# x = [1,2,3]
# # corresponding y axis values
# y = [2,4,1]
 
# # plotting the points 
# plt.plot(x, y)
 
# # naming the x axis
# plt.xlabel('x - axis')
# # naming the y axis
# plt.ylabel('y - axis')
 
# # giving a title to my graph
# plt.title('My first graph!')
 
# # function to show the plot
# plt.show()
import matplotlib.pyplot as plt
import sys

def dataLoad(filename):
  X = []

  file = open(filename)

  lines = file.readlines()

  i = 0
  for line in lines:
    X.append([])
    words = line.split(",")
    for word in words:
      X[i].append(float(word))
    i += 1

  return X


argc = len(sys.argv)

X = dataLoad("LabeledPointsGPU.txt")

x0 = [x[0] for x in X if x[-1] == 0]
y0 = [x[1] for x in X if x[-1] == 0]
x1 = [x[0] for x in X if x[-1] == 1]
y1 = [x[1] for x in X if x[-1] == 1]

plt.scatter(x0, y0, color="r")
plt.scatter(x1, y1)

plt.savefig("../output/{}.png".format(sys.argv[1]), quality=95, format='png',bbox_inches='tight')
plt.show()
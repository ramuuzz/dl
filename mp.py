# 1. Read weights and inputs
print("Enter six weights from input-to-hidden layer:")
w14, w15 = float(input()), float(input())
w24, w25 = float(input()), float(input())
w34, w35 = float(input()), float(input())

print("Enter two hidden-to-output weights:")
w46, w56 = float(input()), float(input())

print("Enter three input values:")
x1, x2, x3 = float(input()), float(input()), float(input())

# 2. Forward pass: weighted sum, then sigmoid
sigmoid = lambda z: 1 / (1 + 2.7 ** (-z))
x4 = sigmoid(x1*w14 + x2*w24 + x3*w34)
x5 = sigmoid(x1*w15 + x2*w25 + x3*w35)
x6 = sigmoid(x4*w46 + x5*w56)

# 3. Output
print("Network output:", x6)

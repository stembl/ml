# functions that implement the Logistic Regression

from scipy import exp, dot, log, array, asarray, linspace, zeros, ones, append
import matplotlib.pyplot as plt

def sigmoid(z):
    '''
    Compute the Sigmoid function

    g = sigmoid(z) returns the sigmoid of z

    z can be a matrix, vector, or scalar
    '''

    g = 1/(1 + exp(-z))

    return(g)

def costLogReg (theta, X, y):
    '''
    Return the cost [J] for a logistic regression.

    theta ~ Hypothesis parameters for the regularized logistic regression
    X     ~ Input values
    y     ~ Output variables or features
    '''

    # m ~ number of training examples
    # n ~ number of features
    [m, n] = X.shape

    # Initialize variables
    J = 0

    # Hypothesis function for logistic regression
    h = sigmoid(dot(X, theta))

    J = (1/m) * (dot(-y.T , log(h)) - dot((1 - y.T) , log(1 - h)))

    return(J)

def costwReg (theta, X, y, lam):
    '''
    Return the cost and gradient [J, grad] for a logistic regression with regularization.

    This function adds the effects of regularization to the results of the costLogReg function.

    theta ~ Hypothesis parameters for the regularized logistic regression
    X     ~ Input values
    y     ~ Output variables or features
    lam   ~ (lambda) Regularization parameter
    '''

    # m ~ number of training examples
    # n ~ number of features
    [m, n] = X.shape

    # Lambda should not be applied to theta_0
    theta_reg = array(theta)
    theta_reg[0] = 0

    J = costLogReg(theta, X, y)

    J = J + lam / (2*m) *dot(theta_reg.T, theta_reg)

    return(J)

def gradLogReg (theta, X, y):
    '''
    Return gradient [grad] for a logistic regression.

    theta ~ Hypothesis parameters for the regularized logistic regression
    X     ~ Input values
    y     ~ Output variables or features
    '''

    # m ~ number of training examples
    # n ~ number of features
    [m, n] = X.shape

    # Initialize variables
    J = 0
    grad = zeros(n)

    # Hypothesis function for logistic regression

    h = sigmoid(dot(X, theta))

    grad = (1/m) * dot(X.T , (h - y))

    return(grad)

def gradwReg (theta, X, y, lam):
    '''
    Return the cost and gradient [J, grad] for a logistic regression with regularization.

    This function adds the effects of regularization to the results of the costLogReg function.

    theta ~ Hypothesis parameters for the regularized logistic regression
    X     ~ Input values
    y     ~ Output variables or features
    lam   ~ (lambda) Regularization parameter
    '''

    # m ~ number of training examples
    # n ~ number of features
    [m, n] = X.shape

    # Lambda should not be applied to theta_0
    theta_reg = array(theta)
    theta_reg[0] = 0

    grad = gradLogReg(theta, X, y)

    grad = grad + (lam * theta_reg / m)

    return(grad)

def plotData(X, y):
    '''
    Plots the datapoints X and y into a new figure.

    Assumes
    * X is an M x 2 matrix
    * y is a series consisting of 1s and 0s
    '''
    X = asarray(X)
    y = asarray(y)

    pos = X[y==1]
    neg = X[y==0]

    plt.figure()
    plt.plot(pos[:,0],pos[:,1], 'yo', label='y = 1')
    plt.plot(neg[:,0],neg[:,1], 'k+', label='y = 0')

def mapFeature(X1, X2, degree=6):
    '''
    Maps the input features to polynomial features

    '''
    mappedX = ones([(X1).size, 1])

    for i in range(degree):
        for j in range(0,i+2):
            mappedX = append(mappedX, (X1**(i+1-j) * X2**j).reshape([X1.size, 1]), axis=1)

    return(mappedX)

def plotDecisionBoundary(theta, X, y, degree=6):

    X0 = X[:,1:3]

    plotData(X0, y)

    if X.shape[1] <= 3:
        print('Error, Need to code this section')

    else:
        # Grid Range
        u = linspace(-1, 1.5, 50)
        v = linspace(-1, 1.5, 50)

        z = zeros([len(u), len(v)])

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = dot(mapFeature(asarray(u[i]), asarray(v[j]), degree), theta)

        # Transpose z before calling contour
        z = z.T

        CS = plt.contour(u, v, z, 0, label='Decision Boundary')
        #plt.clabel(CS, inline=1, fontsize=10)

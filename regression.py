import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


import logging
logging.getLogger().setLevel(logging.DEBUG)

#error function (MSE)
def compute_cost(X, y, w):
    y_hat = np.array(X.dot(w)) 
    j = sum((y_hat - y)**2) / len(y_hat)
    return(j)

#gradient descent
def gradient_descent(X, y, w, epochs, lr):
    cost = np.zeros(epochs)
    for i in range(epochs):

        w = w - (lr / len(X)) * np.sum(X.T.dot( (X.dot(w.T)-y) ), axis=0)
        cost[i] = compute_cost(X, y, w)
    return w, cost


#Carregar dados 
df = pd.read_csv('diamond_prices.csv', header=0)
df = (df - df.mean()) / (df.max() - df.min())

X = df[['carat', 'cut_code', '']]
y = df[['price']]

#X_norm = (X - X.mean()) / (X.max() - X.min())
X_norm = X

mask = np.random.rand(len(df)) < 0.8
X_train = np.array(X_norm[mask])
X_test = np.array(X_norm[~mask])
y_train = np.array(y[mask])[:,0]
y_test = np.array(y[~mask])[:,0]
w = np.ones(np.size(X_train,1))


#print(compute_cost(X_train, y_train, w))

#hyperp
lr = 0.1
reg = 0.01
epochs = 500

#Train
w, cost = gradient_descent(X_train, y_train, w, epochs, lr )
print(w)
print(compute_cost(X_train, y_train, w))

#evaluate
print('erro em validação:' ,compute_cost(X_test, y_test, w))


#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(epochs), cost, 'r')  
ax.set_xlabel('Epochs')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  
plt.show()


#plots
# First construct a grid of (theta0, theta1) parameter pairs and their
# corresponding cost function values.
w0_grid = np.linspace(-1,4,101)
w1_grid = np.linspace(-5,5,101)
J_grid = compute_cost(w0_grid[np.newaxis,:,np.newaxis],
                      w1_grid[:,np.newaxis,np.newaxis])

# A labeled contour plot for the RHS cost function
X, Y = np.meshgrid(w0_grid, w1_grid)
contours = ax[1].contour(X, Y, J_grid, 30)
ax[1].clabel(contours)
# The target parameter values indicated on the cost function contour plot
ax[1].scatter([w[0]]*2,[w[1]]*2,s=[50,10], color=['k','w'])

# Annotate the cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
# Also plot the fit function on the LHS data plot in a matching colour.
colors = ['b', 'g', 'm', 'c', 'orange']
ax[0].plot(x, hypothesis(x, *w[0]), color=colors[0], lw=2,
           label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*w[0]))
for j in range(1,N):
    ax[1].annotate('', xy=theta[j], xytext=theta[j-1],
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
    ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=2,
           label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
ax[1].scatter(*zip(*theta), c=colors, s=40, lw=0)

# Labels, titles and a legend.
ax[1].set_xlabel(r'$\theta_0$')
ax[1].set_ylabel(r'$\theta_1$')
ax[1].set_title('Cost function')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Data and fit')
axbox = ax[0].get_position()
# Position the legend by hand so that it doesn't cover up any of the lines.
ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
             fontsize='small')

plt.show()
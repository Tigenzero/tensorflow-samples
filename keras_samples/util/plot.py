import numpy as np


# Helper Functions
def plot_data(plot, x, y):
    """
    :param plot: Type - matplotlib.pyplot: object which plots the points
    :param x: Type - ndarray: array of x points
    :param y: Type - ndarray: array of y points
    :return: Type - matplotlib.pyplot: plot filled with x,y coordinates and a legend
    """
    # plot class where y==0
    plot.plot(x[y == 0, 0], x[y == 0, 1], 'ob', alpha=0.5)
    # plot class where y==1
    plot.plot(x[y == 1, 0], x[y == 1, 1], 'xr', alpha=0.5)
    plot.legend(['0', '1'])
    return plot


# Common function that draws the decision boundaries
def plot_decision_boundary(model, plot, x, y):
    amin, bmin = x.min(axis=0) - 0.1
    amax, bmax = x.max(axis=0) + 0.1
    # set graph ticks
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make prediction with the model and reshape the output so contourf can plot it
    c = model.predict(ab)
    z = c.reshape(aa.shape)

    plot.figure(figsize=(12, 8))
    # Plot Contour
    plot.contourf(aa, bb, z, cmap='bwr', alpha=0.2)
    # plot the moons of data
    plot_data(plot, x, y)

    return plot
import h5py
import numpy as np
from matplotlib import pyplot as plt

f = h5py.File("./data.hdf", "r")

print(f.keys())

print(f["data"].keys())

xpos = f["data/xpos"][:]
ypos = f["data/ypos"][:]

# plt.scatter(xpos, ypos)
# plt.show()

# Choose a function with parameters f(x, theta),
# where theta is what's being minimized
# Utilize cubic polynomial as chosen function
# def poly_model


# Source - https://stackoverflow.com/a/45274580
# Posted by Chiheb Nexus
# Retrieved 2026-10-27, License - CC BY-SA 3.0


# in this case, args is a list
def cubic(x, args):
    args_len = len(args)
    for k in args:
        yield k * x ** (args_len - 1)
        args_len -= 1


# Source - https://stackoverflow.com/a/45274580
# Posted by Chiheb Nexus
# Retrieved 2026-10-27, License - CC BY-SA 3.0

user_input = ypos  # args
user_var_x = xpos  # x value
somme = sum(cubic(user_var_x, user_input))

print(somme)


# Starting guesses:

a = 
b = 
c = 
d = 
p0 = np.array([a, b, c, d])


# Minimizers:

# Cost function:
# C = Summation(yi - f(xi))**2

# R^2 = Is your model better than the average best fit?
# R^2 = 1- (Summation((yi - f(xi))**2)/ Summation(yi - average(yi))**2)


# Bootstrapping = Bootstrap is a statistical tool to assess the accuracy of an estimate or a model.
# It does so by repeatedly sampling data from the original dataset with replacement.

#example:
def create_bootstrap_samples(sample_size=len(df), n_samples=1000):

    # create a list for sample means
    sample_means = []

    # loop n_samples times
    for i in range(n_samples):

        # create a bootstrap sample of sample_size with replacement
        df_bootstrap_sample = df["charges"].sample(n=sample_size, replace=True)

        # calculate the bootstrap sample mean
        sample_mean = df_bootstrap_sample.mean()

        # add this sample mean to the sample means list
        sample_means.append(sample_mean)

    return pd.Series(sample_means)


# Then, plot distribution

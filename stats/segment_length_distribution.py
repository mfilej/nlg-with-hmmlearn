from sys import stdin, argv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

filename = argv[1]

a = np.loadtxt(filename, dtype=np.int)

num_bins = np.max(a) + 1
if num_bins < 15:
    num_bins = 15

bins = range(num_bins)

plt.hist(a, bins=bins)
plt.title("Segment length distribution")
plt.xlabel("Number of words in segment")
plt.ylabel("Number of occurences")
plt.grid(True)
plt.show()

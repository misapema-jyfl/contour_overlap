"""
Code used to generate a contour plot 
of potassium and sodium solution sets.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Set font for plots
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)


def determine_max_and_min(array):
	"""
	Determines the maximum and minimum value 
	from an NxN array.
	"""
	# return maximum_value, minimum_value
	return max([max(row) for row in array]), min([min(row) for row in array])


def normalize_array(array):
	"""
	Normalizes an NxN array to unity.
	"""
	maximum_value, minimum_value = determine_max_and_min(array)
	if maximum_value == 0:
		print("\nThe overlap is empty! Exiting.")
		sys.exit()
	for i in range(len(array)):
		for j in range(len(array)):	
			array[i][j] = array[i][j]/maximum_value
			
	return array

def get_histogram(df, bins, xmin, xmax, ymin, ymax):
	"""
	Get the 2d histogram of the n, E values
	from the chosen solution set dataframe.

	Applies a Gaussian filter to smooth out 
	the edges of the contour plot.

	Array is normalized by the maximum density
	of the histogram.

	Parameters:
	-----------
	bins : number of bins of the histogram
	xmin, xmax, ymin, ymax : Limits of the histogram
	  (i.e. limits of the dataset)

	Returns:
	--------
	The result dictionary containing the solution set 
	histogram, the 'extent' which can be passed to
	the contour plotter, and the bin edges 'xedges'/'yedges'.
	"""

	x = df["E"]
	y = df["n"]

	rng = [[xmin, xmax],[ymin,ymax]]

	heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=rng, normed=True)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	heatmap = gaussian_filter(heatmap, sigma=1.5)
	heatmap = normalize_array(heatmap)

	result = {}
	result["heatmap"] = heatmap.T
	result["extent"] = extent
	result["xedges"] = xedges
	result["yedges"] = yedges
	
	return result


filenames = {
"k-9+" : "./solution_set_k9+.csv",
"k-10+": "./solution_set_k10+.csv",
"na-7+": "./solution_set_na7+.csv"
}

F_upper = 1E-6
lvl = [.1]	


# Generate the figure
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the outermost (level=.1) contours
# of the solution sets.
df = pd.read_csv(filenames["k-9+"], index_col=None)
c = (df["F"]<F_upper)
df = df[c]
result = get_histogram(df, bins=200, xmin=10, xmax=10000, ymin=1e11, ymax=2.61E12)
k9 = ax.contour(result["heatmap"],extent=result["extent"], levels=lvl, colors="black", origin="lower")

df = pd.read_csv(filenames["k-10+"], index_col=None)
c = (df["F"]<F_upper)
df = df[c]
result = get_histogram(df, bins=200, xmin=10, xmax=10000, ymin=1e11, ymax=2.61E12)
k10 = ax.contour(result["heatmap"],extent=result["extent"], levels=lvl, colors="red", origin="lower")

df = pd.read_csv(filenames["na-7+"], index_col=None)
c = (df["F"]<F_upper)
df = df[c]
result = get_histogram(df, bins=200, xmin=10, xmax=10000, ymin=1e11, ymax=2.61E12)
na = ax.contour(result["heatmap"],extent=result["extent"], levels=lvl, colors="blue", origin="lower")


# Generate proxy labels for the legend
# since the contour plot does not 
# usually expect to be labeled.
proxy = [plt.Rectangle((0,0),1,1,fc = "black"),
plt.Rectangle((0,0),1,1,fc = "red"),
plt.Rectangle((0,0),1,1,fc = "blue")
]
ax.legend(proxy, [r"K$^{9+}$", r"K$^{10+}$", r"Na$^{7+}$"])

# Plot settings
ax.set(xscale="log", yscale="log")
ax.set_xlabel(r"$\left\langle E_e\right\rangle$ (eV)")
ax.set_ylabel(r"$n_e$ (cm$^{-3}$)")
plt.tight_layout()

# Save figure
plt.savefig("contour_overlap.png", dpi=600, format="png")
plt.savefig("contour_overlap.eps", format="eps")
plt.close()
"""
Code used to generate an overlap contour plot of solution sets from the ct-analyzer.
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

def get_histogram(df, bins, xmin, xmax, ymin, ymax, sigma):
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

	if sigma > 0:
		heatmap = gaussian_filter(heatmap, sigma=sigma)
	
	heatmap = normalize_array(heatmap)

	result = {}
	result["heatmap"] = heatmap.T
	result["extent"] = extent
	result["xedges"] = xedges
	result["yedges"] = yedges
	
	return result


def color_cycler(colors):
	"""
	Generator for cycling through named colors in a given list.
	"""
	count = 0
	while True:
		yield colors[count]
		count = (count + 1)%len(colors)

def overlap_contour_plot(filenames, sigma, labels, lvl=[.1], F_upper=1E-6, colors=["k", "r", "b", "m", "orange"]):

	if len(filenames) > len(colors):
		print("Warning: not enough named colors to plot all lines.")

	# Generate the figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Generate proxy labels for the legend
	# since the contour plot does not 
	# usually expect to be labeled.
	proxy = []

	# Color generator
	color_gen = color_cycler(colors=colors)

	# Generate the contours
	for key, arg in filenames.items():
		
		# Get the contour color
		c = next(color_gen)

		# Append the proxy legend
		proxy.append(plt.Rectangle((0,0),1,1, fc=c))

		df = pd.read_csv(arg, index_col=None).query("F < @F_upper")
		result = get_histogram(df=df, bins=200, xmin=10, xmax=10000, ymin=1e11, ymax=2.61E12, sigma=sigma)
		ax.contour(result["heatmap"], extent=result["extent"], levels=lvl, colors=c)

	# Plot settings
	ax.set(xscale="log", yscale="log")
	ax.set_xlabel(r"$\left\langle E_e\right\rangle$ (eV)")
	ax.set_ylabel(r"$n_e$ (cm$^{-3}$)")
	ax.legend(proxy, labels)
	plt.tight_layout()
	
	# Save figure
	plt.savefig("contour_overlap.png", dpi=600, format="png")
	plt.savefig("contour_overlap.eps", format="eps")
	plt.close()


filenames = {
"k-9+" : "./solution_set_k9+.csv",
"k-10+": "./solution_set_k10+.csv",
"na-7+": "./solution_set_na7+.csv"
}

overlap_contour_plot(
	filenames=filenames, 
	sigma=1, 
	labels=[r"K$^{9+}$", r"K$^{10+}$", r"Na$^{7+}$"]
	)




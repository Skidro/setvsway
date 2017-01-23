######################################################################################
# File		: cache.py
# Author	: Waqar Ali
# 
# Description	: This file contains code to simulate a simple set associative cache
######################################################################################

from random import seed, uniform, randint
from os import system
import pylab as pl
import numpy as np
import scipy.stats as stats
import sys

# Create a class for a set associative cache
class Cache(object):
	def __init__(self, sets = 16, ways = 16):
		# Initialize storage for cache blocks
		self.storage = [[0 for x in xrange(ways)] for y in xrange(sets)]

		# Store the specified sets and ways
		self.sets = sets
		self.ways = ways

		# Initialize the random cache utilization metrics
		self.utilization()

		# Initialize counter for misses
		self.misses = 0

	def utilization(self):
		# Generate weights for set utilization
		self.set_weights = [1.0 / self.sets] * self.sets

		for index in xrange(self.sets - 1):
			# Generate a random number in the uniform range
			# between 0 and 1
			number = uniform(0, 1)

			# Adjust partition utilizations based on the generated
			# random number
			factor = number - 0.5
			self.set_weights[index + 1] -= self.set_weights[index] * factor
			self.set_weights[index] += self.set_weights[index] * factor

		# Accumulate set weights to adjust partition utilizations
		for index in range(1, self.sets):
			self.set_weights[index] += self.set_weights[index - 1]

	def set_working_set(self, working_set_size):
		# Set the working set size for the rest of the experiment
		self.working_set_size = working_set_size

		# Adjust partition sizes based on the working set size
		self.part_weights = [int(x * working_set_size) for x in self.set_weights]

	def reference(self, block_number):
		# Find out in which set the block lies
		selected_set = 0
		for weight in self.part_weights:
			if block_number <= weight:
				break
			else:
				selected_set += 1

		# Assess whether the referece is a hit
		hit = False
		for way in xrange(self.ways):
			if self.storage[selected_set][way] == block_number:
				# Mark the reference as a hit
				hit = True
				break

		# Increment the miss-counter if the block was a hit
		if not hit:
			self.misses += 1

			# Randomly chose a cache line to store the cache block
			line = randint(0, self.ways - 1)

			# Store the cache block in the selected line
			self.storage[selected_set][line] = block_number

	def get_misses(self):
		# Return the total number of cache misses encountered
		return self.misses

def test_run(sets = 4, ways = 32):
	# Instantiate the cache class
	cache = Cache(sets, ways)

	# Specify the working set size : 100% Utilization
	working_set_size = sets * ways
	cache.set_working_set(working_set_size)

	# Specify the total number of cache references to generate
	accesses = 100000

	# Generate cache references
	for i in xrange(accesses):
		# Randomly choose a cache block to access
		block_number = randint(0, working_set_size - 1)

		# Make a cache reference to the selected block
		cache.reference(block_number)

	# Calculate the miss-rate
	misses = cache.get_misses()
	miss_rate = (float(misses) / accesses) * 100

	# Return the statistics to caller for this run
	return miss_rate

def plot_data(miss_rate_data, partition_type):
	# Calculate the miss-rate mean and standard deviation
	mr_sorted = sorted(miss_rate_data)
	mr_mean = np.mean(mr_sorted)
	mr_std = np.std(mr_sorted)

	# Keep track of the maxima of miss-rate
	mr_min = mr_sorted[0]
	mr_max = mr_sorted[-1]

	# Fit a normal curve over the miss-rate data
	fit = stats.norm.pdf(mr_sorted, mr_mean, mr_std)

	# Create a figure to save the plot
	fig = pl.figure(1, figsize = (35, 15))

	# Set the figure name
	figname = "MR_" + partition_type.upper() + ".png"

	# Make the miss-rate plot
	pl.plot(mr_sorted, fit)

	# Plot the histograms along the normal curve
	pl.hist(mr_sorted, normed = True, bins = np.arange(mr_min, mr_max, (mr_max - mr_min) / 10))

	# Create a title for this plot
	title = "Miss-Rate PDF for " + partition_type + " Partitioned Cache"
	pl.suptitle(title)
	pl.title('$\mu$ : %.2f%% | [ %.2f%% ] | ( %.2f%% ) | $\sigma$ : %.2f' % (mr_mean, mr_max, mr_min, mr_std))

	# Specify labels for x and y axes
	pl.xlabel("Miss-Rate")
	pl.ylabel("PDF")

	# Set the limit of x-axis
	pl.xlim(0, 100)

	# Hide ticks for the y-axis
	pl.tick_params(axis = 'y', left = 'off', right = 'off', labelleft = 'off')

	# Show the plot
	pl.show()

	# Save the figure
	fig.savefig(figname)
		
def main(argv):
	# Specify the number of test runs to be conducted
	runs = 1000

	# Seed the random number generator with uniform seed value
	seed(runs)

	# Create storage for keeping track of experiment stats
	miss_rate_data = []

	# Specify buffer length for terminal output
	progress_bar_length = 100
	progress_percent = format(0, '3d') + '%'

	# Set the partitioning parameters as per the CLI arguments
	if not argv or argv[0] == "Set":
		# Assume set partitioning
		number_of_sets = 4
		number_of_ways = 16

		# Store the partition type to create appropriate labels
		partition_type = "Set"
	else:
		# Set way partitioning parameters
		number_of_sets = 16
		number_of_ways = 4

		# Store the partition type to create appropriate labels
		partition_type = "Way"

	# Conduct the experiment for the specified runs
	for run in xrange(runs):
		miss_rate = test_run(number_of_sets, number_of_ways)

		# Store the miss-rate in the data array
		miss_rate_data.append(miss_rate)

		# Calculate overall progress of the experiment
		net_progress = int((float(run) / (runs - 1) * 100))
		progress_percent = format(net_progress, '3d') + '%'
		progress = u'\u2588' * net_progress + ' ' * (progress_bar_length - net_progress)

		# Print the progress to terminal
		system('clear')
		print 'Progess : [ ' + progress + ' : ' + progress_percent + ' ]'

	# Plot the miss-rate data
	plot_data(miss_rate_data, partition_type)

# Specify main as the entry point of this program
if __name__ == "__main__":
	# Invoke the main function
	main(sys.argv[1:])

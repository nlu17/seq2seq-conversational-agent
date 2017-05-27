'''
This program plots the lengths of source input and target pairs.

The intention is for one to use this to help determine bucket sizes.

Maybe in the future I will implement a clustering algorithm to autonomously find
bucket sizes
'''

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import util.dataprocessor as data_utils
import sys
import numpy as np
import os
import tensorflow as tf

DATA_DIR = "data/"

num_bins = 50
plot_histograms = True

def main():
    files = [
        os.path.join(DATA_DIR, "train_source.txt"),
        os.path.join(DATA_DIR, "train_target.txt"),
        os.path.join(DATA_DIR, "test_source.txt"),
        os.path.join(DATA_DIR, "test_target.txt")]
    if not (os.path.exists(files[0]) and os.path.exists(files[1])
         and os.path.exists(files[2]) and os.path.exists(files[3])):
        print("Train/Test files not detected, creating now...")
        data_processor = data_utils.DataProcessor(
            max_vocab_size=40000,
            tokenizer_str="basic")
        data_processor.run()

    source_lengths = []
    target_lengths = []
    count = 0
    for i in range(0, len(files), 2):
        source_path = files[i]
        target_path = files[i+1]
        with open(source_path, "r") as source_file:
            with open(target_path, "r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                counter = 0
                while source and target:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    num_source_ids = len(source.split())
                    source_lengths.append(num_source_ids)
                    #plus 1 for EOS token
                    num_target_ids = len(target.split()) + 1
                    target_lengths.append(num_target_ids)
                    source, target = source_file.readline(), target_file.readline()
    if plot_histograms:
        target = plot_histo_lengths("target lengths", target_lengths)
        source = plot_histo_lengths("source lengths", source_lengths)
        plt.legend([source, target], ['Source length', 'Target length'])
    else:
        # Plot scatter.
        plot_scatter_lengths("target vs source length", "source length",
            "target length", source_lengths, target_lengths)
    plt.show()


def plot_scatter_lengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.ylim(0, max(y_lengths))
	plt.xlim(0,max(x_lengths))

def plot_histo_lengths(title, lengths):
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    color = "red" if "target" in title else "green"
    n, bins, patches = plt.hist(x,  num_bins, facecolor=color, alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plot, = plt.plot(bins, y, color=color)
    print(title, bins)
    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
    plt.xlim(0,max(lengths))

    return plot


if __name__=="__main__":
	main()

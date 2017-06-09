import sys

def generate_plots(tuples, title_name):
		import pylab

		K = 40
		from matplotlib import pylab as plt
		from matplotlib.pylab import  xticks
		import math
		import matplotlib.patches as mpatches

		
		y = [-_[0] for _ in tuples[4:K]]
		x = [i for i in range(1, len(y) + 1)]

		pylab.title(title_name)
		pylab.plot(x, y, color='blue', marker='o')
		pylab.show()



file_path = sys.argv[1]



lines = open(file_path, 'r', encoding='utf-8').readlines()


word_count = {}
for line in lines:
	parts = line.strip().split()
	words = parts[:-2]

	for word in words:
		word_count[word] = 1 + word_count.get(word, 0)


tuples = sorted((-word_count[word], word) for word in word_count)


generate_plots(tuples, file_path)
import sys

source_file = 'data/' + sys.argv[1]

lines = open(source_file, 'r', encoding = 'utf-8').readlines()

i = 1 

while i < len(lines):
	parts = lines[i].strip().split()

	bleu_score = float(parts[-2])
	perpl_score = float(parts[-1])

	parts = lines[i-1].strip().split()

	prev_bleu_score = float(parts[-2])
	prev_perpl_score = float(parts[-1])

	print(prev_perpl_score, '\t', perpl_score)
	i += 2


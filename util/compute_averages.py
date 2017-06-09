import numpy as np
import sys

out_file = sys.argv[1]

with open(out_file, "r") as f:
    content = [line.strip().rsplit(" ", 2)[-2:] for line in f.readlines()]
    content = np.array([[float(c[0]), float(c[1])] for c in content])

    bleu_scores = content[:, 0]
    perp_scores = content[:, 1]

    bleu_scores = bleu_scores[~np.isnan(bleu_scores)]
    perp_scores = perp_scores[~np.isnan(perp_scores)]

    bleu_avg = np.average(bleu_scores)
    perp_avg = np.average(perp_scores)

    print("BLEU average:", bleu_avg)
    print("Perplexity average:", perp_avg)

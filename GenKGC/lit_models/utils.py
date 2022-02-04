import json
import numpy as np

def rank_score(ranks):
	# prepare the dataset
	len_samples = len(ranks)
	hits10 = [0] * len_samples
	hits5 = [0] * len_samples
	hits1 = [0] * len_samples
	mrr = []


	for idx, rank in enumerate(ranks):
		if rank <= 10:
			hits10[idx] = 1.
			if rank <= 5:
				hits5[idx] = 1.
				if rank <= 1:
					hits1[idx] = 1.
		mrr.append(1./rank)
	

	return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

def acc(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return (preds == labels).mean()
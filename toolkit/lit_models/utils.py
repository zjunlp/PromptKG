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
import torch.nn as nn
import torch
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
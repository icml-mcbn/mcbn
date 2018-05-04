import sys
import numpy as np
import math

f = open(sys.argv[1], 'r')

num_samples = 10000
num_classes = 10

prob = np.zeros([1, num_classes], dtype=float)
probs = np.zeros([num_samples, num_classes], dtype=float)
labels = np.ones(num_samples, dtype=int)*-1

total = 0
id = 0
for line in f:
    parts = line.split()
    label = int(parts[0])
    if labels[id] != -1 and labels[id] != label:
        raise ValueError('A very specific bad thing happened.')
    if labels[id] == -1:
        labels[id] = label
    for c in range(num_classes):
        prob[0][c] = float(parts[c+2])
    probs[id:id+1][:] += prob
    id = (id+1) % num_samples
    total += 1
    if total % num_samples == 0:
        probs_cur = probs / (total/num_samples)
        tp = 0
        NLL = 0
        for i in range(num_samples):
            pred = np.argmax(probs_cur[i][:])
            NLL -= math.log(probs_cur[i][labels[i]])
            tp += pred==labels[i]
        print('accuracy at epoch #%d: %f' % (total/num_samples,float(tp)/num_samples))
        print('NLL at epoch #%d: %f' % (total/num_samples,NLL))

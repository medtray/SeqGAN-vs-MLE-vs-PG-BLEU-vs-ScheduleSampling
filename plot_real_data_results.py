import numpy as np
from matplotlib import pyplot as plt

track_blue_seqgan=np.load('experiments/track_blue_seqgan2.npy')
track_blue_mle=np.load('experiments_mle/track_blue_mle2.npy',allow_pickle=True)

plt.plot(track_blue_seqgan,label='SeqGAN')
plt.plot(track_blue_mle,label='MLE')

plt.vlines(500, 0, 50, colors='k', linestyles='dashed',linewidth=0.6)
plt.legend(loc="lower right")
plt.xlabel('Epochs')
plt.ylabel('BLEU')
plt.show()
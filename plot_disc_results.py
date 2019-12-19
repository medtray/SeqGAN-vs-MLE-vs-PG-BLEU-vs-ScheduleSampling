import numpy as np
from matplotlib import pyplot as plt

track_loss_1=np.load('perf_dict1.npy',allow_pickle=True)
track_loss_1=track_loss_1[()]
track_loss_2=np.load('perf_dict2.npy',allow_pickle=True)
track_loss_2=track_loss_2[()]



plt.plot(np.log(track_loss_1['true_loss'][:1000]),label='LSTM-based discriminator')
plt.plot(np.log(track_loss_2['true_loss'][:1000]),label='CNN-based discriminator')



plt.vlines(800, 6, 7.5, colors='k', linestyles='dashed',linewidth=0.6)

plt.legend(loc="upper right")
plt.xlabel('Epochs')
plt.ylabel('NLL by oracle')
plt.show()
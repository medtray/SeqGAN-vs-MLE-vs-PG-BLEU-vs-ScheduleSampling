import numpy as np
from matplotlib import pyplot as plt

track_loss_200=np.load('perf_dict200.npy',allow_pickle=True)
track_loss_200=track_loss_200[()]
track_loss_400=np.load('perf_dict400.npy',allow_pickle=True)
track_loss_400=track_loss_400[()]
track_loss_800=np.load('perf_dict800.npy',allow_pickle=True)
track_loss_800=track_loss_800[()]
track_loss_1000=np.load('perf_dict1000.npy',allow_pickle=True)
track_loss_1000=track_loss_1000[()]


plt.plot(np.log(track_loss_200['true_loss']),label='SeqGAN with 200 pretraining epochs')
plt.plot(np.log(track_loss_400['true_loss']),label='SeqGAN with 400 pretraining epochs')
plt.plot(np.log(track_loss_800['true_loss']),label='SeqGAN with 800 pretraining epochs')
plt.plot(np.log(track_loss_1000['true_loss']),label='SeqGAN with 1000 pretraining epochs')

plt.vlines(200, 6, 7.5, colors='k', linestyles='dashed',linewidth=0.6)
plt.vlines(400, 6, 7.5, colors='k', linestyles='dashed',linewidth=0.6)
plt.vlines(800, 6, 7.5, colors='k', linestyles='dashed',linewidth=0.6)
plt.vlines(1000, 6, 7.5, colors='k', linestyles='dashed',linewidth=0.6)
plt.legend(loc="upper right")
plt.xlabel('Epochs')
plt.ylabel('NLL by oracle')
plt.show()
from matplotlib import pyplot as plt
import numpy as np
import random

perf_dict_ss=np.load('perf_dict_ss.npy',allow_pickle=True)
perf_dict_ss=perf_dict_ss[()]

perf_dict_mle=np.load('perf_dict_mle.npy',allow_pickle=True)
perf_dict_mle=perf_dict_mle[()]

perf_dict_seqgan=np.load('perf_dict.npy',allow_pickle=True)
perf_dict_seqgan=perf_dict_seqgan[()]

perf_dict_pgbleu=np.load('perf_dict_pgbleu.npy',allow_pickle=True)
perf_dict_pgbleu=perf_dict_pgbleu[()]

plt.figure(1)
plt.plot(perf_dict_seqgan['true_loss'],'-r')
plt.plot(perf_dict_mle['true_loss'],'-g')
plt.plot(perf_dict_ss['true_loss'],'-b')
plt.plot(perf_dict_pgbleu['true_loss'],'-m')
plt.xlim(800,1400)
plt.ylim(700,1400)

plt.figure(2)
plt.plot(np.log(perf_dict_seqgan['true_loss']),'-r',label='SeqGAN')
plt.plot(np.log(perf_dict_mle['true_loss']),'-g',label='MLE')
plt.plot(np.log(perf_dict_ss['true_loss']),'-b',label='SS')
plt.plot(np.log(perf_dict_pgbleu['true_loss']),'-m',label='PG-BLEU')

plt.vlines(1000, 6, 7.5, colors='k', linestyles='dashed',linewidth=0.6)
plt.legend(loc="upper right")
plt.xlabel('Epochs')
plt.ylabel('NLL by oracle')
plt.show()
print('done')
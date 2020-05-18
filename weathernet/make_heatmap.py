import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)

lr = [1e-2, 1e-3, 1e-4, 1e-5]
bs = [64, 128, 256, 512]

"""
all_test_accuracies = np.zeros((len(lr), len(bs)))
all_test_accuracies[0,0] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_owl17.npy')
all_test_accuracies[0,1] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_owl18.npy')
all_test_accuracies[0,2] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_owl19.npy')
all_test_accuracies[0,3] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_owl20.npy')
all_test_accuracies[1,0] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive13.npy')
all_test_accuracies[1,1] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive14.npy')
all_test_accuracies[1,2] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive15.npy')
all_test_accuracies[1,3] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive17.npy')
all_test_accuracies[2,0] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive26.npy')
all_test_accuracies[2,1] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive27.npy')
all_test_accuracies[2,2] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive28.npy')
all_test_accuracies[2,3] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive29.npy')
all_test_accuracies[3,0] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive31.npy')
all_test_accuracies[3,1] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive34.npy')
all_test_accuracies[3,2] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_beehive35.npy')
all_test_accuracies[3,3] = np.load('figures/heatmaps/separate_values/heatmap_recall_lr_batch_size_owl21.npy')
np.save('figures/heatmaps/heatmap_recall_lr_batch_size.npy', all_test_accuracies)
"""
all_test_accuracies = np.load('figures/heatmaps/heatmap_validation_accuracy_lr_batch_size.npy')

fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_test_accuracies, annot=True, fmt=".2f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Validation accuracy")
ax.set_xlabel('Learning rate')
ax.set_ylabel('Batch size')
ax.set_xticklabels(lr)
ax.set_yticklabels(bs)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_validation_accuracy_lr_batch_size.pdf')
plt.show()

"""
fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_recall, annot=True, fmt=".4f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Recall")
ax.set_xlabel('Activation layer 2')
ax.set_ylabel('Activation layer 1')
ax.set_xticklabels(activation1)
ax.set_yticklabels(activation2)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_recall_activation1_activation2.pdf')
plt.show()
"""


"""
filters = [16, 32, 64, 128, 256]
kernel_size = [3, 6, 9, 12, 24, 48]


all_test_accuracies = np.load('figures/heatmaps/heatmap_test_accuracy_filters_kernel_size_layer2_v2.npy')
all_train_accuracies = np.load('figures/heatmaps/heatmap_train_accuracy_filters_kernel_size_layer2_v2.npy')
all_train_losses = np.load('figures/heatmaps/heatmap_train_loss_filters_kernel_size_layer2_v2.npy')
all_test_losses = np.load('figures/heatmaps/heatmap_test_loss_filters_kernel_size_layer2_v2.npy')
all_recall = np.load('figures/heatmaps/heatmap_recall_filters_kernel_size_layer2_v2.npy')


fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_train_accuracies, annot=True, fmt=".2f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Training accuracy")
ax.set_xlabel('Kernel size')
ax.set_ylabel('Filters')
ax.set_xticklabels(kernel_size)
ax.set_yticklabels(filters)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_train_accuracy_%s_%s_layer2_v2.pdf' %('filters', 'kernel_size'))


fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_test_accuracies, annot=True, fmt=".2f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Validation accuracy")
ax.set_xlabel('Kernel size')
ax.set_ylabel('Filters')
ax.set_xticklabels(kernel_size)
ax.set_yticklabels(filters)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_validation_accuracy_%s_%s_layer2_v2.pdf' %('filters', 'kernel_size'))


fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_recall, annot=True, fmt=".4f", ax=ax,   cmap="viridis")#cmap="YlGnBu")
ax.set_title("Recall")
ax.set_xlabel('Kernel size')
ax.set_ylabel('Filters')
ax.set_xticklabels(kernel_size)
ax.set_yticklabels(filters)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_recall_%s_%s_layer2_v2.pdf' %('filters', 'kernel_size'))

fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_test_losses, annot=True, fmt=".4f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Validation loss")
ax.set_xlabel('Kernel size')
ax.set_ylabel('Filters')
ax.set_xticklabels(kernel_size)
ax.set_yticklabels(filters)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_validation_loss_%s_%s_layer2_v2.pdf' %('filters', 'kernel_size'))


fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_train_losses, annot=True, fmt=".4f", ax=ax,   cmap="viridis")#cmap="YlGnBu")
ax.set_title("Training loss")
ax.set_xlabel('Kernel size')
ax.set_ylabel('Filters')
ax.set_xticklabels(kernel_size)
ax.set_yticklabels(filters)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig('figures/heatmaps/heatmap_train_loss_%s_%s_layer2_v2.pdf' %('filters', 'kernel_size'))
plt.show()
"""


"""
pool_sizes = [2, 3, 6, 9, 12]

all_test_accuracies = np.zeros((1,len(pool_sizes)))
all_train_accuracies = np.zeros((len(pool_sizes),1))
all_test_losses = np.zeros((len(pool_sizes),1))
all_train_losses = np.zeros((len(pool_sizes),1))
all_recall = np.zeros((1, len(pool_sizes)))

all_test_accuracies[:,0] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_poolsize_beehive31.npy')
all_test_accuracies[:,1] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_poolsize_beehive34.npy')
all_test_accuracies[:,2] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_poolsize_beehive35.npy')
all_test_accuracies[:,3] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_poolsize_beehive36.npy')
all_test_accuracies[:,4] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_poolsize_beehive18.npy')

all_train_accuracies[0] = np.load('figures/heatmaps/separate_values/heatmap_train_accuracy_poolsize_beehive31.npy')
all_train_accuracies[1] = np.load('figures/heatmaps/separate_values/heatmap_train_accuracy_poolsize_beehive34.npy')
all_train_accuracies[2] = np.load('figures/heatmaps/separate_values/heatmap_train_accuracy_poolsize_beehive35.npy')
all_train_accuracies[3] = np.load('figures/heatmaps/separate_values/heatmap_train_accuracy_poolsize_beehive36.npy')
all_train_accuracies[4] = np.load('figures/heatmaps/separate_values/heatmap_train_accuracy_poolsize_beehive18.npy')

all_test_losses[0] = np.load('figures/heatmaps/separate_values/heatmap_test_loss_poolsize_beehive31.npy')
all_test_losses[1] = np.load('figures/heatmaps/separate_values/heatmap_test_loss_poolsize_beehive34.npy')
all_test_losses[2] = np.load('figures/heatmaps/separate_values/heatmap_test_loss_poolsize_beehive35.npy')
all_test_losses[3] = np.load('figures/heatmaps/separate_values/heatmap_test_loss_poolsize_beehive36.npy')
all_test_losses[4] = np.load('figures/heatmaps/separate_values/heatmap_test_loss_poolsize_beehive18.npy')

all_train_losses[0] = np.load('figures/heatmaps/separate_values/heatmap_train_loss_poolsize_beehive31.npy')
all_train_losses[1] = np.load('figures/heatmaps/separate_values/heatmap_train_loss_poolsize_beehive34.npy')
all_train_losses[2] = np.load('figures/heatmaps/separate_values/heatmap_train_loss_poolsize_beehive35.npy')
all_train_losses[3] = np.load('figures/heatmaps/separate_values/heatmap_train_loss_poolsize_beehive36.npy')
all_train_losses[4] = np.load('figures/heatmaps/separate_values/heatmap_train_loss_poolsize_beehive18.npy')

all_recall[:,0] = np.load('figures/heatmaps/separate_values/heatmap_recall_poolsize_beehive31.npy')
all_recall[:,1] = np.load('figures/heatmaps/separate_values/heatmap_recall_poolsize_beehive34.npy')
all_recall[:,2] = np.load('figures/heatmaps/separate_values/heatmap_recall_poolsize_beehive35.npy')
all_recall[:,3] = np.load('figures/heatmaps/separate_values/heatmap_recall_poolsize_beehive36.npy')
all_recall[:,4] = np.load('figures/heatmaps/separate_values/heatmap_recall_poolsize_beehive18.npy')



fig, ax = plt.subplots(figsize = (7, 2))
sns.heatmap(all_test_accuracies, annot=True, fmt=".2f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Validation accuracy")
ax.set_xlabel('Pool size')
ax.set_xticklabels(pool_sizes)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#plt.savefig('figures/heatmaps/heatmap_validation_accuracy_%s_%s_layer2.pdf' %('filters', 'kernel_size'))


fig, ax = plt.subplots(figsize = (7, 2))
sns.heatmap(all_recall, annot=True, fmt=".4f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Recall")
ax.set_xlabel('Pool size')
ax.set_xticklabels(pool_sizes)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#plt.savefig('figures/heatmaps/heatmap_validation_accuracy_%s_%s_layer2.pdf' %('filters', 'kernel_size'))
plt.show()
"""

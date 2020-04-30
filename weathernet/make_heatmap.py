import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)

activation1 = ['relu', 'tanh', 'sigmoid']
activation2 = ['relu', 'tanh', 'sigmoid']

all_test_accuracies = np.zeros((len(activation1), len(activation2)))
all_test_accuracies[0,:] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_activation1_activation2_beehive28.npy')
all_test_accuracies[1,:] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_activation1_activation2_beehive29.npy')
#all_test_accuracies[2,:] = np.load('figures/heatmaps/separate_values/heatmap_test_accuracy_activation1_activation2_beehive30.npy')

all_recall = np.zeros((len(activation1), len(activation2)))
all_recall[0,:] = np.load('figures/heatmaps/separate_values/heatmap_recall_activation1_activation2_beehive28.npy')
all_recall[1,:] = np.load('figures/heatmaps/separate_values/heatmap_recall_activation1_activation2_beehive29.npy')
#all_test_accuracies[2,:] = np.load('figures/heatmaps/separate_values/heatmap_recall_activation1_activation2_beehive30.npy')

fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_test_accuracies[:2], annot=True, fmt=".2f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Validation accuracy")
ax.set_xlabel('Activation layer 2')
ax.set_ylabel('Activation layer 1')
ax.set_xticklabels(activation1)
ax.set_yticklabels(activation2)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(all_recall[:2], annot=True, fmt=".4f", ax=ax,  cmap="viridis")# cmap="YlGnBu")
ax.set_title("Recall")
ax.set_xlabel('Activation layer 2')
ax.set_ylabel('Activation layer 1')
ax.set_xticklabels(activation1)
ax.set_yticklabels(activation2)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


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

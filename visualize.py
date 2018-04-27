import matplotlib.pyplot as plt
from sklearn import manifold

def plot_tsne (X, labels):
    
    no_of_samples = 1000
    
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    tsne_y = tsne.fit_transform(X[0:no_of_samples])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_y[:, 0], tsne_y[:, 1], tsne_y[:, 1], c=labels[0:no_of_samples])
    plt.savefig('tsne_vis.pdf', format='pdf')
    plt.show()

def plt_outer_loop (MSE, update_norm):
    
    plt.plot(MSE, label='MSE')
    plt.plot(update_norm, label='Cross Entropy')
    plt.xlabel('Number of iterations')
    plt.legend(bbox_to_anchor=(1.00, 1), loc=1, borderaxespad=0.)
    plt.savefig('Outer_loop.pdf', format='pdf')
    plt.show()
    
def plt_inner_loop (val_error_function,  loss_function, i):
    
    plt.plot(val_error_function, label='Validation error')
    plt.plot(loss_function, label='Loss function')
    plt.ylabel('MSE values for row %d missing value predictions'% i)
    plt.xlabel('Number of epochs')
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    plt.ylim((0,1))
    plt.show()
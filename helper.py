import keras
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from sklearn.preprocessing import PolynomialFeatures


class log_weights(keras.callbacks.Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """
    def __init__(self,get_weights):
        self.get_weights=get_weights     
        self.weights=[]
        
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        #self.epoch.append(epoch)
        self.weights.append(self.get_weights(self.model))

class plot_learning_curve(keras.callbacks.Callback):
    def __init__(self, plot_interval=1, evaluate_interval=10, x_val=None, y_val_categorical=None,epochs=None):
        self.plot_interval = plot_interval
        self.evaluate_interval = evaluate_interval
        self.x_val = x_val
        self.y_val_categorical = y_val_categorical
        self.epochs=epochs
        #self.model = model
    
    def on_train_begin(self, logs={}):
        print('Begin training')
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        if self.evaluate_interval is None:
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))
            self.i += 1
        
        if (epoch%self.plot_interval==0):
            clear_output(wait=True)
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20,5))
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            if self.epochs:
                ax1.set_xlim(-1,self.epochs)
            ax1.legend()

            ax2.plot(self.x, self.acc, label="acc")
            ax2.plot(self.x, self.val_acc, label="val_acc")
            if self.epochs:
                ax2.set_xlim(-1,self.epochs)
            ax2.legend()
            plt.show();


def gen_frame(num, weights_list, w1_mesh, w2_mesh,J,set_weights,ax,model,X,y):
    surf = ax.plot_surface(w1_mesh, w2_mesh, J, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    w1=weights_list[num][0]
    w2=weights_list[num][0]
    set_weights(model,w1,w2)
    j=get_loss(model,X,y)
    ax.scatter(w1, w2, j,c='k')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_title('Función de costo')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.

def animate_loss_surface(X,y,weights_list,model,set_weights, w1_range,w2_range,n_points):
    w1_mesh,w2_mesh,J=plot_loss_surface(X,y,model,set_weights, w1_range,w2_range,n_points,plot=False)
    fig = plt.figure(figsize=(16,10))
    ax = fig.gca(projection='3d')
    loss_ani = animation.FuncAnimation(fig, gen_frame, len(weights_list), fargs=(weights_list, w1_mesh, w2_mesh,J,set_weights,ax,model,X,y),interval=50, blit=True)
    loss_ani.save('loss.mp4')

def plot_loss_surface(X,y,model,set_weights, w1_range,w2_range,n_points,plot=True):
    
    # Make data.
    w1 = np.arange(w1_range[0], w1_range[1], (w1_range[1]-w1_range[0]) / n_points)
    w2 = np.arange(w2_range[0], w2_range[1], (w2_range[1]-w2_range[0]) / n_points)
    w1_mesh,w2_mesh = np.meshgrid(w1, w2)
    J=np.zeros(w1_mesh.shape)
    lista=np.array([w1_mesh,w2_mesh])
    for w1_i,w1_v in enumerate(w1):
        for w2_i,w2_v in enumerate(w2):
            J[w1_i,w2_i]=get_loss(w1_mesh[w1_i,w2_i],w2_mesh[w1_i,w2_i],model,X,y,set_weights)
    if plot:
        # Plot the surface.
        fig = plt.figure(figsize=(16,10))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(w1_mesh, w2_mesh, J, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        #ax.plot(history[0,:],history[1,:],history[2,:])
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_title('Función de costo')
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # rotate the axes and update
        #angle=0
        #ax.view_init(0, angle)
        plt.draw()
        plt.show()
    return w1_mesh,w2_mesh,J

def get_loss(w1,w2,model,X,y,set_weights):
    set_weights(model,w1,w2)
    return model.evaluate(X,y,verbose=0)[0]

def plotBoundary(data, labels, clf_1=None, N=20,degree=False,include_bias=False,ax=None,mins=None,maxs=None):
    class_1 = data[labels == 1]
    class_0 = data[labels == 0]
    N = 300
    if mins == None:
        mins = data[:,:2].min(axis=0)
    if maxs == None:
        maxs = data[:,:2].max(axis=0)
    x1 = np.linspace(mins[0], maxs[0], N)
    x2 = np.linspace(mins[1], maxs[1], N)
    x1, x2 = np.meshgrid(x1, x2)
    X=np.c_[x1.flatten(), x2.flatten()]
    if degree:
        poly=PolynomialFeatures(degree,include_bias=include_bias)
        X=poly.fit_transform(X)
    if ax==None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
    cm = plt.cm.RdBu
    if(clf_1):
        Z_nn = clf_1.predict_proba(X)[:, 0]
        # Put the result into a color plot
        Z_nn = Z_nn.reshape(x1.shape)
        ax.contour(x1, x2, Z_nn, (0.5,), colors='b', linewidths=1)
    ax.scatter(class_1[:,0], class_1[:,1], color='b', s=20, alpha=0.5)
    ax.scatter(class_0[:,0], class_0[:,1], color='r', s=20, alpha=0.5)
    #plt.show()

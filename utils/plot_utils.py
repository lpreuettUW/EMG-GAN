import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses(metrics, epoch, output_dir):
    losses = np.array(metrics)
    plt.plot(losses[:,0], label='Discriminator')
    plt.plot(losses[:,1], label='Generator')
    plt.title('Losses - Generator / Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Metric')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Losses_{}.png'.format(epoch)))
    #plt.show()
    plt.close()
            
    plt.figure()        
    plt.plot(losses[:,2], label='FFT MSE')
    plt.title('FFT of Generated Signal')
    plt.xlabel('Epoch')
    plt.ylabel('FFT MSE')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'FFT_{}.png'.format(epoch)))
    #plt.show()
    plt.close()
    
    plt.figure()        
    plt.plot(losses[:,3], label='DTW Distance')
    plt.title('DTW Distance of Generated Signal')
    plt.xlabel('Epoch')
    plt.ylabel('DTW Distance')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'DTW_{}.png'.format(epoch)))
    #plt.show()
    plt.close()
    
    plt.figure()        
    plt.plot(losses[:,4], label='Cross-correlation')
    plt.title('Cross-correlation between reference signal and generated signal')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-correlation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Cross_correlation_{}.png'.format(epoch)))
    #plt.show()
    plt.close()
        
def plot_prediction(signal, output_dir, kfold, epoch = 0):
    plt.figure()
    axes = plt.gca()
    #axes.set_ylim([-1.0,1.0])
    plt.plot(signal[0], label='Generated Signal')
    plt.xlabel('Epoch' + str(epoch))
    plt.ylabel('EMG')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Generated_' + str(epoch) + '_' + str(kfold) + '.png'))
    #plt.show()
    plt.close()
    
def plot_reference(signal, output_dir, kfold, epoch):
    plt.figure()
    axes = plt.gca()
    #axes.set_ylim([-1.0,1.0])
    plt.plot(signal[0], label='Reference Signal')
    plt.xlabel('Epoch' + str(epoch))
    plt.ylabel('EMG')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Reference_' + str(epoch) + '_' + str(kfold) + '.png'))
    #plt.show()
    plt.close()



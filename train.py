import argparse
#import json
import os
from models.dcgan import DCGAN
from utils.metrics import *
from utils.plot_utils import plot_losses
from utils.data_utils import DataLoader

def train(args):
    sample_interval = 25
    #plot_losses_flag = False

    # Create a DataLoader utility object
    data_loader = DataLoader(args)

    verbose = False

    if verbose:
        print(f'args: {args}')
        print(f'target sample size {int(data_loader.get_target_seq_len())}')

    for kfold_k in range(5):
        print(f'k fold: {kfold_k}')
        data_loader.load_fold(kfold_k)

        # Create a new DCGAN object
        dcgan = DCGAN(args.noise_dim, int(data_loader.get_target_seq_len()), args.output_dir, training=True)

        metrics = []
        best_g_loss = best_dtw = None

        for epoch in range(args.num_epochs):
            data_loader.shuffle()
            for signals in data_loader.get_batches():
                # Adversarial ground truths
                valid = np.ones((signals.shape[0], 1))
                fake = np.zeros_like(valid)

                if verbose:
                    print(f'batch shape {signals.shape}')
                # Generate latent noise for generator
                noise = dcgan.generate_noise(signals)

                if verbose:
                    print(f'noise shape {noise.shape}')

                # Generate a batch of new fake signals and evaluate them against the discriminator
                gen_signal = dcgan.generator.predict(noise)

                if verbose:
                    print(f'gen signal shape {gen_signal.shape}')

                validated = dcgan.critic.predict(gen_signal)

                if verbose:
                    print(f'validated shape {validated.shape}')

                #Sample real and fake signals

                # ---------------------
                #  Calculate metrics
                # ---------------------

                # Calculate metrics on best fake data
                metrics_index = np.argmax(validated)

                #Calculate metrics on first fake data
                #metrics_index = 0

                generated = gen_signal[metrics_index].flatten()
                reference = signals[metrics_index].flatten()

                if verbose:
                    print(f'flattened generated shape {generated.shape}')
                    print(f'flattened reference shape {reference.shape}')

                fft_metric, fft_ref, fft_gen = loss_fft(reference, generated)
                dtw_metric = dtw_distance(reference, generated)
                cc_metric = cross_correlation(reference, generated)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                d_loss_real = dcgan.critic.model.train_on_batch(signals, valid) #train on real data
                d_loss_fake = dcgan.critic.model.train_on_batch(gen_signal, fake) #train on fake data
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real) #mean loss

                # ---------------------
                #  Train Generator
                # ---------------------
                g_loss = dcgan.combined.train_on_batch(noise, valid) #train combined model

                # Plot the progress
                print("%d [D loss: %f, acc: %f] [G loss: %f] [FFT Metric: %f] [DTW Metric: %f] [CC Metric: %f]" % (epoch, d_loss[0], d_loss[1], g_loss, fft_metric, dtw_metric, cc_metric[0]))
                metrics.append([[d_loss[0]], [g_loss], [fft_metric], [dtw_metric], [cc_metric[0]]])

                early_stop = False
                # If at save interval => save generated image samples
                if epoch > 100:
                    if epoch % sample_interval == 0:
                        #if config["save_sample"]:
                        dcgan.save_sample(args.output_dir, epoch, kfold_k, signals)
                        dcgan.save_critic(args.output_dir, epoch, kfold_k)
                        dcgan.save_generator(args.output_dir, epoch, kfold_k)
                    if (best_g_loss is not None and g_loss - best_g_loss > 0.25) or (best_dtw is not None and dtw_metric - best_dtw > 10):
                        print('early stopping because model is not improving')
                        early_stop = True

                    if best_g_loss is None or best_g_loss > g_loss:
                        best_g_loss = g_loss
                    if best_dtw is None or best_dtw > dtw_metric:
                        best_dtw = dtw_metric

                if early_stop:
                    break
            if early_stop:
                break

        #dcgan.save_sample(args.output_dir, epoch, signals)
        #dcgan.save_critic(args.output_dir, kfold_k)
        #dcgan.save_generator(args.output_dir, kfold_k)
        plot_losses(metrics, epoch, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG-GAN - Train')

    parser.add_argument('data_dir', help='input data directory', type=str)
    parser.add_argument('output_dir', help='dir to store all output', type=str)
    #parser.add_argument('--config_json', '-config', default='configuration.json', type=str, help='configuration json file path')
    parser.add_argument('--dataset', type=str, choices=['JL', 'JY', 'LP', 'VB'], default='JL')  # JL JY LP VB
    parser.add_argument('--finger', type=str, choices=['Index', 'Middle', 'Ring', 'Pinky', 'Thumb'])
    parser.add_argument('--gpu', help='set CUDA_VISIBLE_DEVICES environment variable', type=str, default=None)
    parser.add_argument('--batch_size', help='num samps in each batch', type=int, default=128)
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=5000)
    parser.add_argument('--noise_dim', help='number of time steps to generate a synthetic from', type=int, default=200)

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.output_dir = os.path.join(args.output_dir, args.dataset, args.finger)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    #config_file = args.config_json
    # with open(config_file) as json_file:
    #     config = json.load(json_file)

    train(args)

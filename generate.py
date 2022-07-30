import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
from utils.plot_utils import plot_prediction
from utils.plot_utils import plot_reference
from models.dcgan import DCGAN
from utils.data_utils import DataLoader

def generate(args):
    # Create a DataLoader utility object
    data_loader = DataLoader(args)

    for k in range(5):
        data_loader.load_fold(k)

        # Create a new DCGAN object
        dcgan = DCGAN(args.noise_dim, int(data_loader.get_target_seq_len()), args.data_dir)

        # Load existing model from saved_models folder (you can pass different indexes to see the effect on the generated signal)
        dcgan.load(args.data_dir, args.finger, k) #loads the last trained generator

        noise = dcgan.generate_noise(data_loader.train_data)
        generated_signals = dcgan.predict(noise)
        generated_signals = data_loader.unnormalize(generated_signals)
        np.savetxt(os.path.join(args.data_dir, f'generated_signals_k{k}'), generated_signals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG-GAN - Generate EMG signals based on pre-trained model')

    parser.add_argument('data_dir', help='input data directory', type=str)
    parser.add_argument('--dataset', type=str, choices=['JL', 'JY', 'LP', 'VB'], default='JL')  # JL JY LP VB
    parser.add_argument('--finger', type=str, choices=['Index', 'Middle', 'Ring', 'Pinky', 'Thumb'])
    parser.add_argument('--gpu', help='set CUDA_VISIBLE_DEVICES environment variable', type=str, default=None)
    parser.add_argument('--noise_dim', help='number of time steps to generate a synthetic from', type=int, default=200)

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.data_dir = os.path.join(args.data_dir, args.dataset, args.finger)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f'data path DNE: {args.data_dir}')

    generate(args)

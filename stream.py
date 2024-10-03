import argparse
from pathlib import Path

import libemg
import numpy as np

from emg_regression.data_collection import collect_data, Device


def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description='Stream EMG data for visualization or collection.')
    parser.add_argument('device', type=str, choices=('emager', 'myo', 'oymotion', 'sifi'), help='Device to stream. Choices are emager, myo, oymotion, sifi.')
    parser.add_argument('--skip_analyze', action='store_true', help='Flag to skip calling analyze_hardware() method.')
    subparsers = parser.add_subparsers(description='Streaming objectives.', dest='objective', required=True)

    visualize_parser = subparsers.add_parser('visualize', description='Live visualization of EMG signal.')
    visualize_parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to use per plot frame.')
    visualize_subparsers = visualize_parser.add_subparsers(description='Visualization methods.', dest='method', required=True)
    visualize_subparsers.add_parser('heatmap', description='Spatial heatmap visualization.')
    visualize_subparsers.add_parser('time', description='Time series plot with all channels on one plot.')
    channels_parser = visualize_subparsers.add_parser('channels', description='Time series plot with each channels on its own plot.')
    channels_parser.add_argument('channels', type=str, help="Channels to display. Pass as a string of comma-separated values (e.g., '1,2,10').")

    collect_parser = subparsers.add_parser('collect', description='Collect data.')
    collect_parser.add_argument('data_directory', type=str, help='Relative path to data directory (minus movements). Collected data will be stored in data_directory.')
    collect_parser.add_argument('media_directory', type=str, help='Relative path to animation directory (minus movements).')
    collect_parser.add_argument('movements', type=str, help='Motion combinations to perform. List separated by commas. Options are open-close, pro-sup, open-pro-close-sup, close-pro-open-sup.')
    collect_parser.add_argument('num_reps', type=int, help='# of reps.')
    
    args = parser.parse_args()
    print(args)


    device = Device(args.device)
    fs = device.fs
    smi = device.stream()

    online_data_handler = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)
    if not args.skip_analyze:
        online_data_handler.analyze_hardware()
    
    if args.objective == 'visualize':
        fi = libemg.filtering.Filter(fs)
        fi.install_common_filters()
        online_data_handler.install_filter(fi)

        # Visualize
        method = args.method
        if method == 'heatmap':
            remap_function = lambda x: np.reshape(x, (x.shape[0], 4, 16))
            online_data_handler.visualize_heatmap(num_samples=args.num_samples, remap_function=remap_function, feature_list=['MAV', 'RMS'])
        elif method == 'time':
            online_data_handler.visualize(num_samples=args.num_samples, block=True)
        elif method == 'channels':
            channels = args.channels.split(',')
            channels = list(map(int, channels))
            online_data_handler.visualize_channels(channels, num_samples=args.num_samples)
        else:
            raise ValueError(f"Unexpected value for visualization method. Got: {method}.")
    elif args.objective == 'collect':
        # Collect data
        movements = args.movements.replace(' ', '') # remove any spaces
        movements = movements.split(',')
        for movement in movements:
            media_directory = Path(args.media_directory, movement).absolute().as_posix()
            data_directory = Path(args.data_directory, movement).absolute().as_posix()

            collect_data(online_data_handler, media_directory, data_directory, args.num_reps)

    print('Stream finished.')


if __name__ == '__main__':
    main()


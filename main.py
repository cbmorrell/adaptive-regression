import argparse

import libemg
import numpy as np
from utils.data_collection import Device

from experiment import Experiment, Config

def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description='Stream EMG data for visualization or collection.', usage='python main.py sifi subject-001 within-sgt sgt')
    parser.add_argument('device', type=str, choices=('emager', 'myo', 'oymotion', 'sifi'), help='Device to stream. Choices are emager, myo, oymotion, sifi.')
    parser.add_argument('subject_id', type=str, help='Subject ID.')
    parser.add_argument('--analyze', action='store_true', help='Flag to call analyze_hardware() method.')
    subparsers = parser.add_subparsers(description='Experiment stage.', dest='stage', required=True)

    sgt_parser = subparsers.add_parser('sgt', description='Collect data.')
    sgt_parser.add_argument('--visualization_method', default='time', type=str, help='Visualization method before collecting data. Options are heatmap, time, or comma-separated channels (e.g., 4,8,10).')

    subparsers.add_parser('adaptation', description='Perform live adaptation.')
    subparsers.add_parser('validation', description='Perform Fitts task.')

    args = parser.parse_args()
    print(args)

    experiment = Experiment(Config(subject_id=args.subject_id, stage=args.stage, device=Device(args.device)))
    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for sm_item in experiment.shared_memory_items:
        smm.create_variable(*sm_item)

    online_data_handler, p = experiment.setup_live_processing()
    if args.analyze:
        online_data_handler.analyze_hardware()
    
    if args.stage == 'sgt':
        # Visualize
        method = args.visualization_method
        if method == 'heatmap':
            # Assume EMaGer
            remap_function = lambda x: np.reshape(x, (x.shape[0], 4, 16))
            online_data_handler.visualize_heatmap(num_samples=experiment.config.device.fs, remap_function=remap_function, feature_list=['MAV', 'RMS'])
        elif method == 'time':
            online_data_handler.visualize(num_samples=experiment.config.device.fs, block=True)
        else:
            # Passed in list of channels
            channels = method.replace(' ', '').split(',')
            channels = list(map(int, channels))
            online_data_handler.visualize_channels(channels, num_samples=experiment.config.device.fs)

        experiment.start_sgt(online_data_handler)

    else:
        experiment.run_isofitts(online_data_handler)

    print('------------------Main script complete------------------')


if __name__ == '__main__':
    main()


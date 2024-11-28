import argparse

import libemg

from experiment import Experiment, Config

def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description='Stream EMG data for visualization or collection.')
    subparsers = parser.add_subparsers(description='Experiment stage.', dest='objective', required=True)

    config_parser = subparsers.add_parser('config', description='Create configuration file for current condition.')
    config_parser.add_argument('subject_directory', type=str, help='Directory to store data in. Stem will be taken as subject ID.')
    config_parser.add_argument('device', type=str, choices=('emager', 'myo', 'oymotion', 'sifi'), help='Device to stream. Choices are emager, myo, oymotion, sifi.')
    config_parser.add_argument('dominant_hand', type=str, choices=('left', 'right'), help='Dominant hand of participant. Determines direction for Fitts.')
    config_parser.add_argument('condition_idx', type=int, help='Index of current condition (starts at 0).')

    run_parser = subparsers.add_parser('run', description='Collect data.')
    run_parser.add_argument('config_path', type=str, help='Path to configuration file.')
    run_parser.add_argument('stage', type=str, choices=('sgt', 'adaptation', 'validation'), help='Stage of experiment.')
    run_parser.add_argument('--analyze', action='store_true', help='Flag to call analyze_hardware() method.')

    args = parser.parse_args()
    print(args)

    if args.objective == 'config':
        config = Config(args.subject_directory, args.dominant_hand, args.device, args.condition_idx)
        config.save()
        return

    config = Config.load(args.config_path)

    experiment = Experiment(config, args.stage)
    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for sm_item in experiment.shared_memory_items:
        smm.create_variable(*sm_item)

    online_data_handler, _ = experiment.setup_live_processing()
    if args.analyze:
        online_data_handler.analyze_hardware()
    
    if args.stage == 'sgt':
        online_data_handler.visualize(num_samples=experiment.config.device.fs, block=True)
        experiment.start_sgt(online_data_handler)
    else:
        experiment.run_isofitts(online_data_handler)

    print('------------------Main script complete------------------')


if __name__ == '__main__':
    main()


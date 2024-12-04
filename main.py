import argparse

import libemg

from experiment import Experiment, Participant, make_participant, make_config

def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description='Stream EMG data for visualization or collection.')
    subparsers = parser.add_subparsers(description='Experiment stage.', dest='objective', required=True)

    config_parser = subparsers.add_parser('config', description='Create configuration file for current condition.')
    config_parser.add_argument('subject_directory', type=str, help='Directory to store data in. Stem will be taken as subject ID.')
    config_parser.add_argument('device', type=str, choices=('emager', 'myo', 'oymotion', 'sifi'), help='Device to stream. Choices are emager, myo, oymotion, sifi.')
    config_parser.add_argument('dominant_hand', type=str, choices=('left', 'right'), help='Dominant hand of participant. Determines direction for Fitts.')

    run_parser = subparsers.add_parser('run', description='Collect data.')
    run_parser.add_argument('participant', type=str, help='Path to configuration file.')
    run_parser.add_argument('condition_idx', type=int, help='Index of current condition (starts at 0).')
    run_parser.add_argument('stage', type=str, choices=('sgt', 'adaptation', 'validation'), help='Stage of experiment.')
    run_parser.add_argument('--analyze', action='store_true', help='Flag to call analyze_hardware() method.')

    args = parser.parse_args()
    print(args)

    if args.objective == 'config':
        participant = make_participant(args.subject_directory, args.dominant_hand, args.device)
        participant.save()
        return

    participant = Participant.load(args.participant)
    config = make_config(participant, args.condition_idx)

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

    # TODO: Erik also mentioned maybe prompting the user with the rotational fitts interface instead of the cartesian prompt... probably just prompt with bars instead?

    print('------------------Main script complete------------------')


if __name__ == '__main__':
    main()


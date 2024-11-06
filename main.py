import argparse
from pathlib import Path

import libemg
import numpy as np
from utils.adaptation import memory_manager, adapt_manager, AdaptationIsoFitts
from multiprocessing import Process

from config import Config

def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description='Stream EMG data for visualization or collection.', usage='python main.py sifi subject-001 within-sgt sgt')
    parser.add_argument('device', type=str, choices=('emager', 'myo', 'oymotion', 'sifi'), help='Device to stream. Choices are emager, myo, oymotion, sifi.')
    parser.add_argument('subject_id', type=str, help='Subject ID.')
    parser.add_argument('model', type=str, choices=('within-sgt', 'combined-sgt', 'ciil', 'oracle'), help='Model type.')
    parser.add_argument('--analyze', action='store_true', help='Flag to call analyze_hardware() method.')
    subparsers = parser.add_subparsers(description='Streaming objectives.', dest='objective', required=True)

    sgt_parser = subparsers.add_parser('sgt', description='Collect data.')
    sgt_parser.add_argument('--visualization_method', default='time', type=str, help='Visualization method before collecting data. Options are heatmap, time, or comma-separated channels (e.g., 4,8,10).')

    subparsers.add_parser('adaptation', description='Perform live adaptation.')
    subparsers.add_parser('validation', description='Perform Fitts task.')

    args = parser.parse_args()
    print(args)

    config = Config(subject_id=args.subject_id, model=args.model, stage=args.objective, device=args.device)
    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for sm_item in config.shared_memory_items:
        smm.create_variable(*sm_item)

    online_data_handler, p = config.setup_live_processing()
    if args.analyze:
        online_data_handler.analyze_hardware()
    
    if args.objective == 'sgt':
        # Visualize
        method = args.visualization_method
        if method == 'heatmap':
            # Assume EMaGer
            remap_function = lambda x: np.reshape(x, (x.shape[0], 4, 16))
            online_data_handler.visualize_heatmap(num_samples=config.device.fs, remap_function=remap_function, feature_list=['MAV', 'RMS'])
        elif method == 'time':
            online_data_handler.visualize(num_samples=config.device.fs, block=True)
        else:
            # Passed in list of channels
            channels = method.replace(' ', '').split(',')
            channels = list(map(int, channels))
            online_data_handler.visualize_channels(channels, num_samples=config.device.fs)

        config.start_sgt(online_data_handler)

    elif args.objective == 'adaptation':
        mdl = config.setup_online_model(online_data_handler, 'adaptation')

        memoryProcess = Process(target = memory_manager, daemon=True, 
                                args=
                                (
                                    config.DC_data_location,
                                    config.shared_memory_items
                                )
        )
        memoryProcess.start()

        adaptProcess = Process(target = adapt_manager, daemon=True,
                               args=
                               (
                                   config.DC_data_location,
                                   mdl.predictor,
                                   config
                               )
        )
        adaptProcess.start()


        # Create Fitts environment with or without CIIL
        controller = libemg.environments.controllers.RegressorController()
        isofitts = AdaptationIsoFitts(config.shared_memory_items, controller, num_circles=8, num_trials=200, dwell_time=2.0,
                                                           save_file=Path(config.DC_model_file).with_name('AD_fitts.pkl').as_posix())
        isofitts.run()

    elif args.objective == 'validation':
        # assume we have a model from the adaptation phase (whether it was not adapted or adapted)
        mdl = config.setup_online_model(online_data_handler, 'validation')

        # Create Fitts environment
        controller = libemg.environments.controllers.RegressorController()
        isofitts   = libemg.environments.isofitts.IsoFitts(controller, num_circles=8, num_trials=20, dwell_time=1.0,
                                                           save_file=Path(config.DC_model_file).with_name('VAL_fitts.pkl').as_posix())
        isofitts.run()

    print('------------------Main script complete------------------')


if __name__ == '__main__':
    main()


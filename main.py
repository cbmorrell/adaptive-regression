import argparse
from pathlib import Path

import libemg
from libemg.data_handler import RegexFilter, FilePackager
import numpy as np

from config import Config

# -> python stream.py sifi collect Data/subject1000/ images/ abduct-adduct-flexion-extension 5
def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description='Stream EMG data for visualization or collection.', usage='python main.py sifi data/subject1000 sgt images')
    parser.add_argument('device', type=str, choices=('emager', 'myo', 'oymotion', 'sifi'), help='Device to stream. Choices are emager, myo, oymotion, sifi.')
    parser.add_argument('subject_id', type=str, help='Subject ID.')
    parser.add_argument('model', type=str, choices=('within-sgt', 'combined-sgt', 'within-ciil', 'combined-ciil'), help='Model type. Choices are within-sgt, combined-sgt, within-ciil, and combined-ciil.')
    parser.add_argument('--analyze', action='store_true', help='Flag to call analyze_hardware() method.')
    subparsers = parser.add_subparsers(description='Streaming objectives.', dest='objective', required=True)

    sgt_parser = subparsers.add_parser('sgt', description='Collect data.')
    sgt_parser.add_argument('media_directory', type=str, help='Relative path to animation directory (minus movements).')
    sgt_parser.add_argument('--visualization_method', default='time', type=str, help='Visualization method before collecting data. Options are heatmap, time, or comma-separated channels (e.g., 4,8,10).')

    fitts_parser = subparsers.add_parser('fitts', description='Perform Fitts task.')
    fitts_parser.add_argument('--feature_set', type=str, default='MSWT', choices=libemg.feature_extractor.FeatureExtractor.get_feature_groups().keys(), help='Feature set.')

    # Thinking of moving this to a main.py file instead of separate ones
    # Have two stages: sgt and fitts
    # If it's sgt then we visualize automatically
    # Use config methods (e.g., setup_live_processing)
    
    args = parser.parse_args()
    print(args)

    config = Config(subject_id=args.subject_id, model=args.model, stage='sgt')

    online_data_handler = config.setup_live_processing()
    if args.analyze:
        online_data_handler.analyze_hardware()
    
    if args.objective == 'sgt':
        # Visualize
        method = args.visualization_method
        if method == 'heatmap':
            remap_function = lambda x: np.reshape(x, (x.shape[0], 4, 16))
            online_data_handler.visualize_heatmap(num_samples=args.num_samples, remap_function=remap_function, feature_list=['MAV', 'RMS'])
        elif method == 'time':
            online_data_handler.visualize(num_samples=args.num_samples, block=True)
        else:
            # Passed in list of channels
            channels = method.replace(' ', '').split(',')
            channels = list(map(int, channels))
            online_data_handler.visualize_channels(channels, num_samples=args.num_samples)

        config.start_sgt()

    elif args.objective == 'fitts':
        # Add filter
        fi = libemg.filtering.Filter(config.device.fs)
        fi.install_common_filters()
        online_data_handler.install_filter(fi)

        # Parse data
        odh = libemg.data_handler.OfflineDataHandler()
        regex_filters = (
            RegexFilter(left_bound='/', right_bound='/', values=[args.subject_id], description='subject'),
            RegexFilter(left_bound='C_0_R_', right_bound='_emg.csv', values=[str(idx) for idx in range(config.DC_reps)], description='reps')
        )
        package_function = lambda x, y: Path(x).parent.absolute().as_posix() == Path(y).parent.absolute().as_posix()
        metadata_fetchers = (
            FilePackager(RegexFilter('/', '.txt', values=['collection'], description='labels'), package_function),
        )
        data_directory = Path(args.data_directory).absolute().as_posix()
        odh.get_data(data_directory, regex_filters, metadata_fetchers)

        # Model setup
        config.prepare_model_from_sgt() # fits model and stores as .pkl
        config.setup_model()

        # Create Fitts environment
        controller = libemg.environments.controllers.RegressorController()
        isofitts = libemg.environments.isofitts.IsoFitts(controller, num_circles=8, num_trials=20, dwell_time=1.0,
                                                         save_file=Path(config.DC_model_file).with_name('fitts.pkl').as_posix())
        isofitts.run()

    print('------------------Main script complete------------------')


if __name__ == '__main__':
    main()


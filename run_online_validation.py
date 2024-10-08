from pathlib import Path
from argparse import ArgumentParser

import libemg
from libemg.data_handler import RegexFilter, FilePackager
from libemg.environments.isofitts import IsoFitts
from libemg.environments.controllers import RegressorController
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from utils.data_collection import Device

def main():
    parser = ArgumentParser(description='Run regression model online.')
    parser.add_argument('device', type=str, choices=('emager', 'myo', 'sifi', 'oymotion'), help='EMG device to stream.')
    parser.add_argument('model', type=str, choices=('svm', 'linear'), help='Model type.')
    parser.add_argument('data_directory', type=str, help='Relative path to data directory (including subject #).')

    subparsers = parser.add_subparsers(description='Validation method.', dest='validation', required=True, help='Validation types.')
    fitts_parser = subparsers.add_parser('fitts')
    subparsers.add_parser('visualize')
    fitts_parser.add_argument('--num_circles', default=8, help='# of circles in Fitts test.')
    fitts_parser.add_argument('--num_trials', default=20, help='# of trials.')
    args = parser.parse_args()
    print(args)

    subject = args.data_directory.split('/')[-1]
    odh = libemg.data_handler.OfflineDataHandler()
    regex_filters = (
        RegexFilter(left_bound='/', right_bound='/', values=[subject], description='subject'),
        RegexFilter(left_bound='/', right_bound='/', values=['open-close', 'pro-sup'], description='movement'),
        RegexFilter(left_bound='C_0_R_', right_bound='_emg.csv', values=[str(idx) for idx in range(5)], description='reps')
    )
    package_function = lambda x, y: Path(x).parent.absolute().as_posix() == Path(y).parent.absolute().as_posix()
    metadata_fetchers = (
        FilePackager(RegexFilter('/', '.txt', values=['labels'], description='labels'), package_function),
    )
    data_directory = Path(args.data_directory).absolute().as_posix()
    odh.get_data(data_directory, regex_filters, metadata_fetchers)

    device = Device(args.device)
    fi = libemg.filtering.Filter(device.fs)
    fi.install_common_filters()
    fi.filter(odh)

    window_size_ms = 150
    window_inc_ms = 40
    window_size_samples = device.fs * window_size_ms / 1000
    window_inc_samples = device.fs * window_inc_ms / 1000
    windows, metadata = odh.parse_windows(window_size_samples, window_inc_samples, metadata_operations=dict(labels=lambda x: x[-1]))
    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = fe.get_feature_groups()['HTD']
    features = fe.extract_features(feature_list, windows, array=True)
    labels = metadata['labels']
    
    if args.model == 'svm':
        model = MultiOutputRegressor(SVR())
    elif args.model == 'linear':
        model = MultiOutputRegressor(LinearRegression())
    else:
        raise ValueError(f"Unexpected value for model. Got: {args.model}.")
    model.fit(features, labels)

    smi = device.stream()
    online_data_handler = libemg.data_handler.OnlineDataHandler(smi)
    online_data_handler.install_filter(fi)

    offline_regressor = libemg.emg_predictor.EMGRegressor(model)
    online_regressor = libemg.emg_predictor.OnlineEMGRegressor(offline_regressor, window_size_samples, window_inc_samples, online_data_handler, feature_list, std_out=False, smm=False)
    online_regressor.run(block=False)

    if args.validation == 'fitts':
        controller = RegressorController()
        prediction_map = None
        fitts = IsoFitts(controller, num_circles=args.num_circles, num_trials=args.num_trials, save_file=Path(data_directory, 'fitts.pkl').absolute().as_posix(), prediction_map=prediction_map)
        fitts.run()
    elif args.validation == 'visualize':
        online_regressor.visualize(legend=True)
    else:
        raise ValueError(f"Unexpected value for validation. Got: {args.validation}.")

    print('Online validation complete!')
    
    
if __name__ == '__main__':
    main()

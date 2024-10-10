from pathlib import Path
from argparse import ArgumentParser

from libemg.data_handler import RegexFilter, FilePackager
import libemg
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from libemg.environments.isofitts import IsoFitts
from libemg.environments.controllers import RegressorController

# -> python run_online_validation.py linear Data/subject1000 fitts/visualize
def main():
    parser = ArgumentParser(description='Run regression model online.')
    parser.add_argument('model', type=str, choices=('svm', 'linear'), help='Model type.')
    parser.add_argument('data_directory', type=str, help='Relative path to data directory (including subject #).')
    parser.add_argument('--skip_analyze', action='store_true', help='Flag to skip analyzing the online predictor.')

    subparsers = parser.add_subparsers(description='Validation method.', dest='validation', required=True, help='Validation types.')
    fitts_parser = subparsers.add_parser('fitts')
    subparsers.add_parser('visualize')
    fitts_parser.add_argument('--num_circles', default=8, help='# of circles in Fitts test.')
    fitts_parser.add_argument('--num_trials', default=20, help='# of trials.')
    args = parser.parse_args()
    print(args)

    subject = args.data_directory.split('/')[-1]
    print(subject)
    odh = libemg.data_handler.OfflineDataHandler()
    regex_filters = (
        RegexFilter(left_bound='/', right_bound='/', values=[subject], description='subject'),
        RegexFilter(left_bound='/', right_bound='/', values=['open-close', 'pro-sup', 'abduct-adduct-flexion-extension'], description='movement'),
        RegexFilter(left_bound='C_0_R_', right_bound='_emg.csv', values=[str(idx) for idx in range(5)], description='reps')
    )
    package_function = lambda x, y: Path(x).parent.absolute().as_posix() == Path(y).parent.absolute().as_posix()
    metadata_fetchers = (
        FilePackager(RegexFilter('/', '.txt', values=['labels'], description='labels'), package_function),
    )
    data_directory = Path(args.data_directory).absolute().as_posix()
    odh.get_data(data_directory, regex_filters, metadata_fetchers)


    if odh.data[0].shape[1] == 8:
        # Assume myo data
        fs = 1500
        window_size = 200
        window_inc = 50
        _, smi = libemg.streamers.sifi_bioarmband_streamer(name="BioPoint_v1_1",
                                                           ppg=False,
                                                           eda=False,
                                                           imu=False,
                                                           ecg=False)
    else:
        # Assume EMaGer data
        fs = 1010
        window_size = 150
        window_inc = 40
        _, smi = libemg.streamers.emager_streamer()

    fi = libemg.filtering.Filter(fs)
    fi.install_common_filters()
    fi.filter(odh)

    windows, metadata = odh.parse_windows(window_size, window_inc, metadata_operations=dict(labels=lambda x: x[-1]))
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

    online_data_handler = libemg.data_handler.OnlineDataHandler(smi)
    online_data_handler.install_filter(fi)

    offline_regressor = libemg.emg_predictor.EMGRegressor(model)
    offline_regressor.add_deadband(0.2)
    online_regressor = libemg.emg_predictor.OnlineEMGRegressor(offline_regressor, window_size, window_inc, online_data_handler, feature_list, std_out=True, smm=False)
    online_regressor.run(block=False)
    # if not args.skip_analyze:
    #     online_regressor.analyze_predictor()

    if args.validation == 'fitts':
        controller = RegressorController()
        controller.start()
        fitts = IsoFitts(controller=controller, num_circles=args.num_circles, num_trials=args.num_trials, save_file=Path(data_directory, 'fitts.pkl').absolute().as_posix())
        fitts.run()
        pass
    elif args.validation == 'visualize':
        online_regressor.visualize(legend=True)
    else:
        raise ValueError(f"Unexpected value for validation. Got: {args.validation}.")

    print('Online validation complete!')
    
    
if __name__ == '__main__':
    main()

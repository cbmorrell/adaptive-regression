from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
import wandb
from libemg.data_handler import RegexFilter, FilePackager
import libemg
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from emg_regression.fitts import FittsLawTest
from emg_regression.logging import Logger
from emg_regression.parsing import read_pickle, write_pickle
from emg_regression.preprocessing import EMGDataset
from emg_regression.data_collection import Device

from analyze_data import get_reps, Config, log_results

ENCODER = 'Transformer'
FEATURE_SET = 'HTD'


def train(args):
    reps = get_reps(args.path, dtype=int)
    train_reps = reps[:-2]
    val_reps = reps[-2:-1]  # this leaves one rep for testing offline data... need to see what we think of that... should I use all data when training the online model?
    test_reps = reps[-1:]

    config = Config(args.path, args.model, args.feature_set)
    odh = config.make_odh()
    logger = Logger(project='self-supervised-exploration', config=config, tags=['train_online_model'])
    torch.set_default_device(args.device)

    with logger:
        odh_test = odh.isolate_data('reps', test_reps)
        if args.model == 'linear':
            odh_train = odh.isolate_data('reps', train_reps + val_reps) # no validation set needed
            pipeline = config.make_data_pipeline(odh_train)
            inputs_train, labels_train = pipeline(odh_train)
            inputs_test, labels_test = pipeline(odh_test)
            model = MultiOutputRegressor(LinearRegression())
            model.fit(inputs_train, labels_train)
            dataset_val = None
        else:
            odh_train = odh.isolate_data('reps', train_reps)
            odh_val = odh.isolate_data('reps', val_reps)
            pipeline = config.make_data_pipeline(odh_train)
            inputs_train, labels_train = pipeline(odh_train)
            inputs_val, labels_val = pipeline(odh_val)
            inputs_test, labels_test = pipeline(odh_test)
            model = config.make_model(inputs_train.shape[-1], labels_train.shape[1])

            dataloader_train = config.make_dataloader(inputs_train, labels_train)
            dataloader_val = config.make_dataloader(inputs_val, labels_val)
            dataset_val = dataloader_val.dataset
            model.fit(dataloader_train, num_epochs=config['num_epochs'], logger=logger, validation_dataloader=dataloader_val)

        dataset_test = config.make_dataloader(inputs_test, labels_test).dataset
        log_results(model, dataset_test, logger, pipeline, args.device, artifact_name=f"{config.subject}-{config.model}-online", validation_dataset=dataset_val)

    print('Training complete.')

def fitts(args):
    api = wandb.Api()
    artifact = api.artifact(f"cmorrell/self-supervised-exploration/{args.artifact}")
    artifact_path = Path(artifact.download()).absolute()
    run_config = artifact.logged_by().config
    config = Config.from_run_config(run_config)
    print(config)
    pipeline = read_pickle(artifact_path.joinpath('pipeline.pkl').as_posix())


    if config.model == 'linear':
        model_path = artifact_path.joinpath('model.pkl').absolute().as_posix()
        model = read_pickle(model_path)
    else:
        model = config.make_model(pipeline.input_size, pipeline.num_dofs)
        model_path = artifact_path.joinpath('model.pt').absolute().as_posix()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    print(f"Loaded model from {model_path}.")

    device = Device(config.device)
    smi = device.stream()
    online_data_handler = libemg.data_handler.OnlineDataHandler(smi)
    online_data_handler.install_filter(pipeline.filter)

    offline_regressor = libemg.emg_predictor.EMGRegressor(model, deadband_threshold=0.15)
    online_regressor = libemg.emg_predictor.OnlineEMGRegressor(offline_regressor, config.window_size, config.window_increment,
                                                                online_data_handler, pipeline.feature_list, std_out=False, smm=False,
                                                                feature_queue_length=config.context_length)
    online_regressor.install_standardization(pipeline.scaler)
    online_regressor.run(block=False)

    if args.visualize:
        online_regressor.visualize(legend=True)

    savefile = Path('fitts.pkl').absolute()
    fitts = FittsLawTest(num_circles=10, num_trials=20, savefile=savefile.as_posix())
    fitts.run()

    logger = Logger(project='self-supervised-exploration', config=config, tags=['fitts'])
    with logger:
        artifact = wandb.Artifact(name=f"{config.subject}-{config.model}", type='fitts')
        logger.log_artifact(artifact)

    savefile.unlink()   # clean up Fitts file
    print('Online validation complete.')


def main():
    parser = ArgumentParser(description='Run regression model online.')
    subparsers = parser.add_subparsers(title='Online analyses', required=True, dest='analysis')

    train_parser = subparsers.add_parser('train', help='Train model for online use.')
    train_parser.add_argument('device', type=str, help='PyTorch device to load tensors on.')
    train_parser.add_argument('path', type=str, help='Path to participant data directory.')
    train_parser.add_argument('model', type=str, choices=('contrastive-transformer', 'baseline-transformer', 'linear'), help='Model type.')
    train_parser.add_argument('feature_set', type=str, choices=('HTD', 'MSWT'), help='Feature group to use.')

    fitts_parser = subparsers.add_parser('fitts', help='Run Fitts test.')
    fitts_parser.add_argument('artifact', type=str, help='Artifact name.')
    fitts_parser.add_argument('--visualize', action='store_true', help='Flag to add visualization of regressor output before Fitts test.')
    args = parser.parse_args()
    print(args)

    if args.analysis == 'train':
        train(args)
    elif args.analysis == 'fitts':
        fitts(args)
    else:
        raise ValueError(f"Unexpected value for analysis. Got: {args.analysis}.")
    
    
if __name__ == '__main__':
    main()

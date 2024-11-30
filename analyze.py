from pathlib import Path
import math
import pickle
from argparse import ArgumentParser

import libemg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from libemg.environments.controllers import RegressorController

from experiment import Config, MODELS


RESULTS_PATH = Path('results')
DPI = 600


def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        file_data = pickle.load(f)
    return file_data


def format_model_names(models):
    def format_name(name):
        return name.replace('-', ' ').title().replace('Sgt', 'SGT')

    if isinstance(models, str):
        return format_name(models)

    return [format_name(model) for model in models]


def get_config(participant, model):
    config_file = [file for file in Path('data').rglob('config.json') if participant in file.as_posix() and model in file.as_posix()]
    assert len(config_file) == 1, f"Expected a single matching config file, but got {config_file}."
    return Config.load(config_file[0])


def is_unfinished_trial(run_log, trial_mask):
    cursor = run_log['cursor_position'][trial_mask[-1]]
    target = run_log['goal_target'][trial_mask[-1]]
    return not in_target(cursor, target)


def in_target(cursor, target):
    return math.dist(cursor[:2], target[:2]) < target[2]/2 + cursor[2]/2


def calculate_efficiency(run_log, trial_mask):
    distance_travelled = np.sum([math.dist(run_log['cursor_position'][trial_mask[i]][0:2], run_log['cursor_position'][trial_mask[i-1]][0:2]) for i in range(1,len(trial_mask))])
    fastest_path = math.dist((run_log['cursor_position'][trial_mask[0]])[0:2], (run_log['goal_target'][trial_mask[0]])[0:2])
    return fastest_path / distance_travelled


def calculate_throughput(run_log, trial_mask):
    trial_time = run_log['global_clock'][trial_mask[-1]] - run_log['global_clock'][trial_mask[0]]
    starting_cursor_position = (run_log['cursor_position'][trial_mask[0]])[0:2]
    target = run_log['goal_target'][trial_mask[0]]
    target_position = target[:2]
    target_width = target[-1]
    distance = math.dist(starting_cursor_position, target_position)
    id = math.log2(distance / target_width + 1)
    return id / trial_time


def calculate_overshoots(run_log, trial_mask):
    cursor_locs = np.array(run_log['cursor_position'])[trial_mask]
    targets = np.array(run_log['goal_target'])[trial_mask]
    in_bounds = [in_target(cursor_locs[i], targets[i]) for i in range(0,len(cursor_locs))]
    overshoots = 0
    for i in range(1,len(in_bounds)):
        if in_bounds[i-1] == True and in_bounds[i] == False:
            overshoots += 1 
    return overshoots


def is_timeout_trial(run_log, trial_mask):
    trial_time = run_log['global_clock'][trial_mask[-1]] - run_log['global_clock'][trial_mask[0]]
    return trial_time > 30


def extract_traces(run_log):
    trials = np.unique(run_log['trial_number'])
    traces = []
    for t in trials:
        trial_mask = np.where(run_log['trial_number'] == t)[0]
        traces.append(np.array(run_log['cursor_position'])[trial_mask][:, :2])
    return traces


def extract_model_predictions(run_log):
    controller = RegressorController()
    trials = np.unique(run_log['trial_number'])
    predictions = []
    for t in trials:
        trial_mask = np.where(run_log['trial_number'] == t)[0]
        model_outputs = np.array(run_log['class_label'])[trial_mask]
        model_outputs = [controller._parse_predictions(model_output) for model_output in model_outputs]
        predictions.append(model_outputs)
    return predictions


def extract_fitts_metrics(run_log):
    fitts_results = {
        'timeouts': [],
        'overshoots': [],
        'efficiency': [],
        'throughput': []
    }
    
    trials = np.unique(run_log['trial_number'])
    for t in trials:
        trial_mask = np.where(run_log['trial_number'] == t)[0]
        if len(trial_mask) <= 1 or is_unfinished_trial(run_log, trial_mask):
            continue
        if is_timeout_trial(run_log, trial_mask):
            # Ignore trial
            fitts_results['timeouts'].append(t)
            continue
        fitts_results['throughput'].append(calculate_throughput(run_log, trial_mask))
        fitts_results['overshoots'].append(calculate_overshoots(run_log, trial_mask))
        fitts_results['efficiency'].append(calculate_efficiency(run_log, trial_mask))
        
    summary_metrics = dict.fromkeys(fitts_results.keys())
    summary_metrics['timeouts'] = fitts_results['timeouts']
    summary_metrics['overshoots'] = np.sum(fitts_results['overshoots'])
    summary_metrics['efficiency'] = np.mean(fitts_results['efficiency'])
    summary_metrics['throughput'] = np.mean(fitts_results['throughput'])
    return summary_metrics


def plot_pilot_distance_vs_proportional_control():
    # NOTE: Used to determine appropriate distance to proportional control mapping during piloting
    with open('/Users/cmorrell/Code/adaptive-regression/data/subject-003/ciil/val_fitts.pkl', 'rb') as f:
        ciil_data = pickle.load(f)

    predictions = np.linalg.norm(ciil_data['current_direction'], axis=1) / 25
    target_positions = np.array(ciil_data['goal_target'])[:, :2]
    cursor_positions = np.array(ciil_data['cursor_position'])[:, :2]
    distances = np.linalg.norm(target_positions - cursor_positions, axis=1)

    plt.figure()
    plt.scatter(distances, predictions)


def plot_fitts_metrics(participants):
    throughputs = []
    efficiencies = []
    overshoots = []
    for model in MODELS:
        model_throughputs = []
        model_efficiencies = []
        model_overshoots = []
        for participant in participants:
            config = get_config(participant, model)
            run_log = read_pickle_file(config.validation_fitts_file)
            fitts_metrics = extract_fitts_metrics(run_log)
            model_throughputs.append(fitts_metrics['throughput'])
            model_efficiencies.append(fitts_metrics['efficiency'])
            model_overshoots.append(fitts_metrics['overshoots'])
        throughputs.append(np.mean(model_throughputs))
        efficiencies.append(np.mean(model_efficiencies))
        overshoots.append(np.mean(model_overshoots))


    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 6), layout='constrained')
    models = format_model_names(MODELS)
    axs[0].bar(models, throughputs)
    axs[1].bar(models, efficiencies)
    axs[2].bar(models, overshoots)

    axs[0].set_ylabel('Throughput')
    axs[1].set_ylabel('Path Efficiency')
    axs[2].set_ylabel('Overshoots')
    title = 'Usability Metrics'
    fig.suptitle(title)
    if len(participants) == 1:
        # Only analyzing 1 participant - add their ID to title
        fig.suptitle(f"{title} ({participants[0]})")
    else:
        fig.savefig(RESULTS_PATH.joinpath('fitts-metrics.png'), dpi=DPI)


def plot_fitts_traces(participants):
    fig, axs = plt.subplots(nrows=1, ncols=len(MODELS), figsize=(10, 10))
    cmap = mpl.colormaps['Dark2']
    for model, ax in zip(MODELS, axs):
        for participant_idx, participant in enumerate(participants):
            config = get_config(participant, model)
            run_log = read_pickle_file(config.validation_fitts_file)
            traces = extract_traces(run_log)
            for trace in traces:
                ax.plot(trace[:, 0], trace[:, 1], c=cmap(participant_idx))
            
        ax.set_title(format_model_names(model))
        ax.set_xlabel('X (Pixels)')
        ax.set_ylabel('Y (Pixels)')
    
    title = 'Fitts Traces'
    fig.suptitle(title)
    if len(participants) == 1:
        # Only analyzing 1 participant - add their ID to title
        fig.suptitle(f"{title} ({participants[0]})")
    else:
        fig.savefig(RESULTS_PATH.joinpath('fitts-traces.png'), dpi=DPI)


def plot_simultaneity(participants):
    ...


def plot_action_interference(participants):
    ...


def main():
    parser = ArgumentParser(prog='Analyze offline data.')
    parser.add_argument('-p', '--participants', default=None, help='List of participants to evaluate.')
    args = parser.parse_args()
    print(args)

    if args.participants is None:
        regex_filter = libemg.data_handler.RegexFilter('subject-', right_bound='/', values=[str(idx + 1).zfill(3) for idx in range(100)], description='')
        matching_directories = regex_filter.get_matching_files([path.as_posix() + '/' for path in Path('data').glob('*')])
        participants = [Path(participant).stem for participant in matching_directories]
    else:
        participants = str(args.participants).replace(' ', '').split(',')

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    plot_fitts_metrics(participants)
    plot_fitts_traces(participants)
    
    # TODO: Look at simultaneity, action interference, and usability metrics over time
    plt.show()
    print('-------------Analyze complete!-------------')


if __name__ == '__main__':
    main()

from pathlib import Path
import math
import pickle
from argparse import ArgumentParser

import libemg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from utils.adaptation import TIMEOUT
from experiment import Config, MODELS


DPI = 600


class Plotter:
    def __init__(self, participants, analysis):
        self.participants = participants
        self.analysis = analysis
        self.models = (MODELS[3], MODELS[1], MODELS[2], MODELS[0])  # reorder based on best visual for plots

        if self.analysis == 'within':
            self.exclude_within_dof_trials = False
            self.exclude_combined_dof_trials = True
        elif self.analysis == 'combined':
            self.exclude_within_dof_trials = True
            self.exclude_combined_dof_trials = False
        else:
            self.exclude_within_dof_trials = False
            self.exclude_combined_dof_trials = False
        self.exclude_timeout_trials = False

        self.results_path = Path('results', self.analysis)
        self.results_path.mkdir(parents=True, exist_ok=True)

    def read_log(self, participant, model):
        config_file = [file for file in Path('data').rglob('config.json') if participant in file.as_posix() and model in file.as_posix()]
        assert len(config_file) == 1, f"Expected a single matching config file, but got {config_file}."
        config = Config.load(config_file[0])
        run_log = read_pickle_file(config.validation_fitts_file)

        filter_mask = []
        for t in np.unique(run_log['trial_number']):
            trial_mask = np.where(run_log['trial_number'] == t)[0]
            is_timeout_trial, is_unfinished_trial, is_within_dof_trial = get_trial_flags(run_log, trial_mask)
            if participant == 'subject-002' and model == 'within-sgt' and t == 0:
                # Handling edge case for first trial being a timeout
                is_unfinished_trial = False
                is_timeout_trial = True

            if (is_timeout_trial and self.exclude_timeout_trials) or (is_within_dof_trial and self.exclude_within_dof_trials) or (not is_within_dof_trial 
                and self.exclude_combined_dof_trials) or is_unfinished_trial:
                print(f"Skipping {participant} {model} trial {t}.")
                continue

            filter_mask.extend(trial_mask)
            
        # Return run_log just with correct trials
        for key in run_log.keys():
            run_log[key] = np.array(run_log[key])[filter_mask]

        return run_log
    
    def _plot_fitts_metrics(self):
        def plot_metric_over_time(values, ax, color):
            values = moving_average(values)
            ax.scatter(np.arange(len(values)), values, color=color, s=4)
            ax.plot(values, alpha=0.5, color=color, linestyle='--')

        fig = plt.figure(layout='constrained', figsize=(18, 6))
        cmap = mpl.colormaps['Dark2']
        subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[1, 3])
        bar_axs = subfigs[0].subplots(nrows=3, ncols=1, sharex=True)
        time_axs = subfigs[1].subplots(nrows=3, ncols=len(self.models), sharex=True)
        lines = []
        bar_labels = []
        mean_throughputs = []
        mean_efficiencies = []
        mean_overshoots = []
        zorder = 2  # zorder=2 so points are plotted on top of bar plot
        for model_idx, model in enumerate(self.models):
            model_throughputs = []
            model_efficiencies = []
            model_overshoots = []
            bar_labels.append(format_model_names(model))
            for participant_idx, participant in enumerate(self.participants):
                run_log = self.read_log(participant, model)
                fitts_metrics = extract_fitts_metrics(run_log)
                model_throughputs.append(fitts_metrics['throughput'])   # need to grab metrics over time here
                model_efficiencies.append(fitts_metrics['efficiency'])
                model_overshoots.append(fitts_metrics['overshoots'])

                # Bar plot
                color = cmap(participant_idx)
                lines.append(bar_axs[0].scatter(bar_labels[-1], model_throughputs[-1], label=participant, zorder=zorder, color=color))
                bar_axs[1].scatter(bar_labels[-1], model_efficiencies[-1], zorder=zorder, color=color)
                bar_axs[2].scatter(bar_labels[-1], model_overshoots[-1], zorder=zorder, color=color)

                # Time series plots in a row share y-axis
                time_axs[0, model_idx].sharey(time_axs[0, 0])
                time_axs[1, model_idx].sharey(time_axs[1, 0])
                time_axs[2, model_idx].sharey(time_axs[2, 0])

                # Plot over time
                time_axs[0, model_idx].set_title(bar_labels[-1])
                time_axs[2, model_idx].set_xlabel('Trial #')
                plot_metric_over_time(fitts_metrics['throughput_over_time'], time_axs[0, model_idx], color)
                plot_metric_over_time(fitts_metrics['efficiency_over_time'], time_axs[1, model_idx], color)
                plot_metric_over_time(fitts_metrics['overshoots_over_time'], time_axs[2, model_idx], color)


            mean_throughputs.append(np.mean(model_throughputs))
            mean_efficiencies.append(np.mean(model_efficiencies))
            mean_overshoots.append(np.mean(model_overshoots))

        handles = get_unique_legend_handles(lines)
        bar_color = 'black'
        bar_axs[0].bar(bar_labels, mean_throughputs, color=bar_color)
        bar_axs[1].bar(bar_labels, mean_efficiencies, color=bar_color)
        bar_axs[2].bar(bar_labels, mean_overshoots, color=bar_color)
        bar_axs[0].set_ylabel('Throughput')
        bar_axs[1].set_ylabel('Path Efficiency')
        bar_axs[2].set_ylabel('Overshoots')
        bar_axs[0].set_title('Across Subjects')

        if len(self.participants) != 1:
            # bar_axs[0].set_ylim((0, np.max(mean_throughputs) + 0.4))
            # bar_axs[0].legend(handles.values(), handles.keys(), loc='upper center', ncols=2)
            fig.legend(handles.values(), handles.keys(), loc='upper left', ncols=4)

        return fig

    def _plot_fitts_traces(self):
        fig, axs = plt.subplots(nrows=1, ncols=len(self.models), figsize=(14, 8), layout='constrained', sharex=True, sharey=True)
        cmap = mpl.colormaps['Dark2']
        lines = []
        for model, ax in zip(self.models, axs):
            for participant_idx, participant in enumerate(self.participants):
                run_log = self.read_log(participant, model)
                traces = extract_traces(run_log)
                for trace in traces:
                    lines.extend(ax.plot(trace[:, 0], trace[:, 1], c=cmap(participant_idx), label=participant, linewidth=1, alpha=0.5))
                
            ax.set_title(format_model_names(model))
            ax.set_xlabel('X (Pixels)')
        
        axs[0].set_ylabel('Y (Pixels)')
        if len(self.participants) != 1:
            legend_handles = get_unique_legend_handles(lines)
            axs[-1].legend(legend_handles.values(), legend_handles.keys())

        return fig

    def _plot_dof_activation_heatmap(self):
        # Create heatmap where x is DOF 1 and y is DOF2
        fig = plt.figure(figsize=(16, 4), constrained_layout=True)
        outer_grid = fig.add_gridspec(nrows=1, ncols=len(self.models))
        width_ratios = [2, 1]
        height_ratios = [1, 2]
        # num_bins = 40
        num_bins = 10
        for model_idx, model in enumerate(self.models):
            model_predictions = []
            for participant in self.participants:
                run_log = self.read_log(participant, model)
                predictions = extract_model_predictions(run_log)
                predictions = np.concatenate(predictions)
                model_predictions.append(predictions)
            model_predictions = np.concatenate(model_predictions)

            # Format heatmap + histogram axes
            inner_grid = outer_grid[model_idx].subgridspec(nrows=2, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios)
            axs = inner_grid.subplots()
            heatmap_ax = axs[1, 0]
            x_hist_ax = axs[0, 0]
            y_hist_ax = axs[1, 1]
            axs[0, 1].set_axis_off()    # hide unused axis

            # Plot
            value_range = np.array([[-1, 1], [-1, 1]])
            x_predictions = model_predictions[:, 0]
            y_predictions = model_predictions[:, 1]
            _, _, _, heatmap = heatmap_ax.hist2d(x_predictions, y_predictions, bins=num_bins, range=value_range)
            x_counts, _, _ = x_hist_ax.hist(x_predictions, bins=num_bins, range=value_range[0])
            y_counts, _, _ = y_hist_ax.hist(y_predictions, bins=num_bins, range=value_range[0], orientation='horizontal')
            x_hist_ax.yaxis.set_major_formatter(PercentFormatter(xmax=sum(x_counts), decimals=0))
            y_hist_ax.xaxis.set_major_formatter(PercentFormatter(xmax=sum(y_counts), decimals=0))

            # Formatting
            fig.colorbar(heatmap, ax=heatmap_ax, format=PercentFormatter(xmax=sum(x_counts) + sum(y_counts), decimals=1))
            x_hist_ax.set_title(format_model_names(model))
            heatmap_ax.set_xlabel('DoF Activation (Open / Close)')
            if model == self.models[0]:
                heatmap_ax.set_ylabel('DoF Activation (Pro / Supination)')
        
        return fig

    def plot(self, plot_type):
        if plot_type == 'fitts-metrics':
            fig = self._plot_fitts_metrics()
        elif plot_type == 'fitts-traces':
            fig = self._plot_fitts_traces()
        elif plot_type == 'heatmap':
            fig = self._plot_dof_activation_heatmap()
        else:
            raise ValueError(f"Unexpected value for plot_type. Got: {plot_type}.")

        title = f"{plot_type} {self.analysis} Trials".replace('-', ' ').title()
        fig.suptitle(title)

        filename = plot_type
        if len(self.participants) == 1:
            filename += f"-{self.participants[0]}"
            title += f"({self.participants[0]})"
        filename += '.png'
        filepath = Path(self.results_path, filename)
        fig.savefig(filepath, dpi=DPI)
        print(f"File saved to {filepath.as_posix()}.")


def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        file_data = pickle.load(f)
    return file_data


def moving_average(a, window_size = 3):
    ma = np.cumsum(a)
    ma[window_size:] = ma[window_size:] - ma[:-window_size]
    return ma[window_size - 1:] / window_size


def get_unique_legend_handles(lines):
    unique_lines = {}
    for line in lines:
        label = line.get_label()
        if label not in unique_lines.keys():
            unique_lines[label] = line

    unique_lines = {label: unique_lines[label] for label in sorted(unique_lines.keys())}
    return unique_lines


def format_model_names(models):
    def format_name(name):
        return name.replace('-', ' ').title().replace('Sgt', 'SGT').replace('Ciil', 'CIIL')

    if isinstance(models, str):
        return format_name(models)

    return [format_name(model) for model in models]


def get_trial_flags(run_log, trial_mask):
    initial_cursor_location = run_log['cursor_position'][trial_mask[0]][:2]
    final_cursor = run_log['cursor_position'][trial_mask[-1]]
    target = run_log['goal_target'][trial_mask[-1]]
    trial_time = run_log['global_clock'][trial_mask[-1]] - run_log['global_clock'][trial_mask[0]]
    exceeds_timeout = round(trial_time, 1) >= TIMEOUT # account for rounding...
    cursor_in_target = in_target(final_cursor, target)
    is_timeout_trial = exceeds_timeout and not cursor_in_target
    is_unfinished_trial = (not exceeds_timeout and not cursor_in_target) or len(trial_mask) <= 1
    is_within_dof_trial = np.any(np.abs(np.array(target[:2]) - np.array(initial_cursor_location)) <= target[2] // 2)
    return is_timeout_trial, is_unfinished_trial, is_within_dof_trial


def in_target(cursor, target):
    return math.dist(cursor[:2], target[:2]) < (target[2]/2 + cursor[2]/2)


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


def extract_traces(run_log):
    trials = np.unique(run_log['trial_number'])
    traces = []
    for t in trials:
        trial_mask = np.where(run_log['trial_number'] == t)[0]
        traces.append(np.array(run_log['cursor_position'])[trial_mask][:, :2])
    return traces


def extract_model_predictions(run_log):
    trials = np.unique(run_log['trial_number'])
    predictions = []
    for t in trials:
        trial_mask = np.where(run_log['trial_number'] == t)[0]
        model_outputs = np.array(run_log['class_label'])[trial_mask]
        model_outputs = [list(map(float, model_output.replace('[', '').replace(']', '').split(','))) for model_output in model_outputs]
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
        # if t == trials[-1]:
        #     # Skip the last trial for now - the pilot data is_uninfinished_trial doesn't work b/c of an edge case where timer starts before first timestamp so trial time isn't larger than timeout.
        #     # This isn't needed if we change validation to be X trials always.
        #     # TODO: When we're not looking at pilot data we should change this.
        #     print(f"Skipping {t}")
        #     continue
        # is_timeout_trial, is_unfinished_trial = get_trial_flags(run_log, trial_mask)

        # if is_unfinished_trial:
        #     # Skip trials
        #     print(f"Skipping {t}")
        #     continue

        # if is_timeout_trial:
        #     fitts_results['timeouts'].append(t)

        fitts_results['throughput'].append(calculate_throughput(run_log, trial_mask))
        fitts_results['overshoots'].append(calculate_overshoots(run_log, trial_mask))
        fitts_results['efficiency'].append(calculate_efficiency(run_log, trial_mask))
        
    summary_metrics = {}
    summary_metrics['throughput_over_time'] = fitts_results['throughput']
    summary_metrics['efficiency_over_time'] = fitts_results['efficiency']
    summary_metrics['overshoots_over_time'] = fitts_results['overshoots']
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


def main():
    parser = ArgumentParser(prog='Analyze offline data.')
    parser.add_argument('-p', '--participants', default=None, help='List of participants to evaluate.')
    parser.add_argument('-a', '--analysis', default='all', choices=('all', 'combined', 'within'), help='Subset of data to perform analysis on.')
    args = parser.parse_args()
    print(args)

    sns.set_theme(style='ticks', palette='Dark2')

    if args.participants is None:
        regex_filter = libemg.data_handler.RegexFilter('subject-', right_bound='/', values=[str(idx + 1).zfill(3) for idx in range(100)], description='')
        matching_directories = regex_filter.get_matching_files([path.as_posix() + '/' for path in Path('data').glob('*')])
        participants = [Path(participant).stem for participant in matching_directories]
        participants.sort()
    else:
        participants = str(args.participants).replace(' ', '').split(',')

    plotter = Plotter(participants, args.analysis)
    plotter.plot('fitts-metrics')
    plotter.plot('fitts-traces')
    plotter.plot('heatmap')
    
    # TODO: Look at simultaneity, action interference, and usability metrics over time
    plt.show()
    print('-------------Analyze complete!-------------')


if __name__ == '__main__':
    main()

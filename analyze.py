from pathlib import Path
import math
import pickle
from argparse import ArgumentParser

import pandas as pd
import libemg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter, NullLocator
import seaborn as sns

from utils.adaptation import TIMEOUT, DWELL_TIME
from experiment import MODELS, Participant, make_config, ADAPTIVE_MODELS


class Plotter:
    def __init__(self, participants, analysis, dpi = 400, stage = 'validation'):
        self.participants = participants
        self.analysis = analysis
        self.dpi = dpi
        self.stage = stage
        self.plot_adaptation = self.stage == 'adaptation'
        if self.plot_adaptation:
            self.models = (ADAPTIVE_MODELS[1], ADAPTIVE_MODELS[0])  # reorder based on best visual for plots (oracle, ciil)
        else:
            self.models = (MODELS[3], MODELS[1], MODELS[2], MODELS[0])  # reorder based on best visual for plots (within, combined, oracle, ciil)

        self.results_path = Path('results', self.analysis, self.stage)
        self.results_path.mkdir(parents=True, exist_ok=True)

    def read_participant(self, participant_id):
        participant_files = [file for file in Path('data').rglob('participant.json') if participant_id in file.as_posix() and 'archive' not in file.as_posix()]
        assert len(participant_files) == 1, f"Expected a single matching participant file for {participant_id}, but got {participant_files}."
        return Participant.load(participant_files[0])

    def read_log(self, participant_id, model):
        participant = self.read_participant(participant_id)
        config = make_config(participant, model)
        if self.plot_adaptation:
            fitts_file = config.adaptation_fitts_file
        else:
            fitts_file = config.validation_fitts_file

        return Log(fitts_file, self.analysis)

    def _plot_fitts_metrics_over_time(self):
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
        subject_throughputs = []
        subject_efficiencies = []
        subject_overshoots = []
        zorder = 2  # zorder=2 so points are plotted on top of bar plot
        for model_idx, model in enumerate(self.models):
            model_throughputs = []
            model_efficiencies = []
            model_overshoots = []
            bar_labels.append(format_names(model))
            for participant_idx, participant in enumerate(self.participants):
                log = self.read_log(participant, model)
                fitts_metrics = log.extract_fitts_metrics()
                model_throughputs.append(np.mean(fitts_metrics['throughput']))
                model_efficiencies.append(np.mean(fitts_metrics['efficiency']))
                model_overshoots.append(np.sum(fitts_metrics['overshoots']))

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
                plot_metric_over_time(fitts_metrics['throughput'], time_axs[0, model_idx], color)
                plot_metric_over_time(fitts_metrics['efficiency'], time_axs[1, model_idx], color)
                plot_metric_over_time(fitts_metrics['overshoots'], time_axs[2, model_idx], color)


            subject_throughputs.append(np.mean(model_throughputs))
            subject_efficiencies.append(np.mean(model_efficiencies))
            subject_overshoots.append(np.mean(model_overshoots))

        handles = get_unique_legend_handles(lines)
        bar_color = 'black'
        bar_axs[0].bar(bar_labels, subject_throughputs, color=bar_color)
        bar_axs[1].bar(bar_labels, subject_efficiencies, color=bar_color)
        bar_axs[2].bar(bar_labels, subject_overshoots, color=bar_color)
        bar_axs[0].set_ylabel('Throughput')
        bar_axs[1].set_ylabel('Path Efficiency')
        bar_axs[2].set_ylabel('Overshoots')
        bar_axs[0].set_title('Across Subjects')

        if len(self.participants) != 1:
            time_axs[-1, -1].legend(handles.values(), format_names(handles.keys()), ncols=2)

        return fig

    def _plot_fitts_metrics(self):
        metrics = ['Throughput', 'Path Efficiency', 'Overshoots', '# Trials', 'Completion Rate']
        fig, axs = plt.subplots(nrows=1, ncols=len(metrics), layout='constrained', figsize=(14, 5))
        adaptive_labels = []
        model_labels = []
        throughputs = []
        efficiencies = []
        overshoots = []
        num_trials = []
        completion_rates = []
        for model in self.models:
            for participant in self.participants:
                log = self.read_log(participant, model)
                model_labels.append(format_names(model))
                adaptive_labels.append('Yes' if model in ADAPTIVE_MODELS else 'No')
                fitts_metrics = log.extract_fitts_metrics()
                throughputs.append(np.mean(fitts_metrics['throughput']))
                efficiencies.append(np.mean(fitts_metrics['efficiency']))
                overshoots.append(np.sum(fitts_metrics['overshoots']))
                num_trials.append(fitts_metrics['num_trials'])
                completion_rates.append(fitts_metrics['completion_rate'])

        df = pd.DataFrame({
            'Model': model_labels,
            'Adaptive': adaptive_labels,
            metrics[0]: throughputs,
            metrics[1]: efficiencies,
            metrics[2]: overshoots,
            metrics[3]: num_trials,
            metrics[4]: completion_rates
        })
        for metric, ax in zip(metrics, axs):
            legend = 'auto' if metric == metrics[-1] else False # only plot legend on last axis
            sns.boxplot(df, x='Model', y=metric, ax=ax, hue='Adaptive', legend=legend) # maybe color boxes based on intended and unintended RMSE?
        
        return fig

    def _plot_fitts_traces(self):
        fig, axs = plt.subplots(nrows=1, ncols=len(self.models), figsize=(14, 8), layout='constrained', sharex=True, sharey=True)
        cmap = mpl.colormaps['Dark2']
        lines = []
        for model, ax in zip(self.models, axs):
            for participant_idx, participant in enumerate(self.participants):
                log = self.read_log(participant, model)
                traces = [trial.traces for trial in log.trials]
                for trace in traces:
                    lines.extend(ax.plot(trace[:, 0], trace[:, 1], c=cmap(participant_idx), label=participant, linewidth=1, alpha=0.5))
                
            ax.set_title(format_names(model))
            ax.set_xlabel('X (Pixels)')
        
        axs[0].set_ylabel('Y (Pixels)')
        if len(self.participants) != 1:
            legend_handles = get_unique_legend_handles(lines)
            axs[-1].legend(legend_handles.values(), format_names(legend_handles.keys()))

        return fig

    def _plot_dof_activation_heatmap(self):
        # Create heatmap where x is DOF 1 and y is DOF2
        fig = plt.figure(figsize=(20, 5), constrained_layout=True)
        outer_grid = fig.add_gridspec(nrows=1, ncols=len(self.models))
        width_ratios = [2, 1]
        height_ratios = [1, 2]
        bins = np.round(np.arange(-1.1, 1.2, 0.2), 2)  # extend past 1 to include 1 in arange
        ticks = np.arange(bins.shape[0])
        for model_idx, model in enumerate(self.models):
            model_predictions = []
            for participant in self.participants:
                log = self.read_log(participant, model)
                predictions = [trial.predictions for trial in log.trials]
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
            # nm_mask = np.all(np.abs(model_predictions) < bins[bins.shape[0] // 2], axis=1)
            # model_predictions = model_predictions[~nm_mask]
            x_predictions = model_predictions[:, 0]
            y_predictions = model_predictions[:, 1]
            x_y_counts, _, _ = np.histogram2d(x_predictions, y_predictions, bins=bins, density=False)   # density sets it to return pdf, not % occurrences
            x_hist_ax.hist(x_predictions, bins=bins)
            y_hist_ax.hist(y_predictions, bins=bins, orientation='horizontal')
            heatmap = sns.heatmap(x_y_counts, ax=heatmap_ax, cmap=sns.light_palette('seagreen', as_cmap=True), norm=mpl.colors.LogNorm())

            # Formatting
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.set_minor_locator(NullLocator())  # disable minor (logarithmic) ticks
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=x_predictions.shape[0], decimals=1))
            heatmap_ax.set_xticks(ticks, bins, rotation=90)
            heatmap_ax.set_yticks(ticks, bins, rotation=0)
            heatmap_ax.set_xlabel('Open / Close Activation')
            heatmap_ax.set_ylabel('Pro / Supination Activation')
            x_hist_ax.set_xticks(bins, bins, rotation=90)
            y_hist_ax.set_yticks(bins, bins, rotation=0)
            x_hist_ax.set_title(format_names(model))
            x_hist_ax.set_ylabel('Frequency')
            y_hist_ax.set_xlabel('Frequency')
            x_hist_ax.set_yscale('log')
            y_hist_ax.set_xscale('log')

            # TODO: Maybe annotate with simultaneity metric
        
        return fig

    def _plot_losses(self):
        fig, ax = plt.subplots()
        epochs = []
        losses = []
        model_labels = []
        for model in self.models:
            for participant in self.participants:
                config = make_config(self.read_participant(participant), model)
                loss_df = pd.read_csv(config.loss_file, header=None)
                batch_timestamps = loss_df.iloc[:, 0]
                batch_losses = loss_df.iloc[:, 1]
                epoch_timestamps = np.unique(batch_timestamps)
                epoch_losses = [np.mean(batch_losses[batch_timestamps == timestamp]) for timestamp in epoch_timestamps]
                epochs.extend(list(range(len(epoch_timestamps))))
                losses.extend(epoch_losses)
                model_labels.extend(format_names([model for _ in range(len(epoch_timestamps))]))

        df = pd.DataFrame({
            'Epochs': epochs,
            'Loss': losses,
            'Model': model_labels
        })
        sns.lineplot(df, x='Epochs', y='Loss', hue='Model', ax=ax)
        return fig

    def plot(self, plot_type):
        if plot_type == 'fitts-metrics':
            fig = self._plot_fitts_metrics()
        elif plot_type == 'fitts-metrics-over-time':
            fig = self._plot_fitts_metrics_over_time()
        elif plot_type == 'fitts-traces':
            fig = self._plot_fitts_traces()
        elif plot_type == 'heatmap':
            fig = self._plot_dof_activation_heatmap()
        elif plot_type == 'loss':
            fig = self._plot_losses()
        else:
            raise ValueError(f"Unexpected value for plot_type. Got: {plot_type}.")

        title = f"{plot_type} {self.analysis} Trials".replace('-', ' ').title()
        filename = plot_type
        if len(self.participants) == 1:
            filename += f"-{self.participants[0]}"
            title += f" ({self.participants[0]})"
        fig.suptitle(title)
        filename += '.png'
        filepath = Path(self.results_path, filename)
        fig.savefig(filepath, dpi=self.dpi)
        print(f"File saved to {filepath.as_posix()}.")


class Log:
    def __init__(self, path, analysis) -> None:
        self.path = path
        self.analysis = analysis
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

        with open(self.path, 'rb') as f:
            run_log = pickle.load(f)

        # Using trial mask to get relevant samples - convert to arrays for convenience
        for key in run_log.keys():
            run_log[key] = np.array(run_log[key])

        all_trials = [Trial(run_log, idx) for idx in np.unique(run_log['trial_number'])[:-1]] # skip the final trial b/c it is always incomplete

        trials = []
        for t in all_trials:
            if (t.is_timeout_trial and self.exclude_timeout_trials) or (t.is_within_dof_trial and self.exclude_within_dof_trials) or (not t.is_within_dof_trial 
                and self.exclude_combined_dof_trials):
                print(f"Skipping trial {t.trial_idx} in log {self.path}.")
                continue

            trials.append(t)
            
        self.trials = trials

    def extract_fitts_metrics(self):
        fitts_results = {
            'timeouts': [],
            'overshoots': [],
            'efficiency': [],
            'throughput': [],
            'num_trials': -1,
            'completion_rate': -1
        }
        
        for t in self.trials:

            if t.is_timeout_trial:
                fitts_results['timeouts'].append(t.trial_idx)
            fitts_results['throughput'].append(t.calculate_throughput())
            fitts_results['overshoots'].append(t.calculate_overshoots())
            fitts_results['efficiency'].append(t.calculate_efficiency())

        fitts_results['num_trials'] = len(fitts_results['throughput'])
        fitts_results['completion_rate'] = 1 - (len(fitts_results['timeouts']) / fitts_results['num_trials'])
            
        return fitts_results


class Trial:
    def __init__(self, run_log, trial_idx):
        self.trial_idx = trial_idx

        trial_mask = np.where(run_log['trial_number'] == self.trial_idx)[0]
        self.cursor_positions = run_log['cursor_position'][trial_mask]
        self.targets = run_log['goal_target'][trial_mask]
        self.timestamps = run_log['global_clock'][trial_mask]
        self.traces = self.cursor_positions[:, :2]
        model_output = run_log['class_label'][trial_mask]
        self.predictions = [list(map(float, model_output.replace('[', '').replace(']', '').split(','))) for model_output in model_output]

        initial_cursor = np.array(self.cursor_positions[0])
        target = np.array(self.targets[-1])
        self.trial_time = self.timestamps[-1] - self.timestamps[0]
        self.is_timeout_trial = self.trial_time >= (TIMEOUT * 0.98)   # account for rounding errors
        self.is_within_dof_trial = np.any(np.abs(target[:2] - initial_cursor[:2]) <= (target[2] // 2 + initial_cursor[2] // 2))
        # TODO: Fix is_within_dof_trial... gives the wrong result b/c it determines based on if the first cursor location is in line with the target, which is incorrect if the previous trial was a timeout and the cursor wasn't in the old target.

    @staticmethod
    def in_target(cursor, target):
        return math.dist(cursor[:2], target[:2]) < (target[2]/2 + cursor[2]/2)

    def calculate_efficiency(self):
        distance_travelled = np.sum([math.dist(self.cursor_positions[i][0:2], self.cursor_positions[i-1][0:2]) for i in range(1, len(self.cursor_positions))])
        fastest_path = math.dist((self.cursor_positions[0])[0:2], (self.targets[0])[0:2])
        return fastest_path / distance_travelled

    def calculate_throughput(self):
        trial_time = self.trial_time
        if not self.is_timeout_trial:
            # Only subtract dwell time for successful trials
            trial_time -= DWELL_TIME
        starting_cursor_position = (self.cursor_positions[0])[0:2]
        target = self.targets[0]
        target_position = target[:2]
        target_width = target[-1]
        distance = math.dist(starting_cursor_position, target_position)
        id = math.log2(distance / target_width + 1)
        return id / trial_time

    def calculate_overshoots(self):
        in_bounds = [self.in_target(self.cursor_positions[i], self.targets[i]) for i in range(0, len(self.cursor_positions))]
        overshoots = 0
        for i in range(1,len(in_bounds)):
            if in_bounds[i-1] == True and in_bounds[i] == False:
                overshoots += 1 
        return overshoots


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


def format_names(models):
    def format_name(name):
        replacements = (
            ('Sgt', 'SGT'),
            ('Ciil', 'CIIL'),
            ('Within', 'W'),
            ('Combined', 'C'),
            ('Subject-', 'S')
        )
        formatted_name = name.title()
        for replacement in replacements:
            formatted_name = formatted_name.replace(replacement[0], replacement[1])
        return formatted_name

    if isinstance(models, str):
        return format_name(models)

    return [format_name(model) for model in models]


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
    parser.add_argument('-p', '--participants', default='all', help='List of participants to evaluate.')
    parser.add_argument('-a', '--analysis', default='all', choices=('all', 'combined', 'within'), help='Subset of tasks to perform analysis on.')
    parser.add_argument('-s', '--stage', default='validation', choices=('adaptation', 'validation'), help='Stage to analyze.')
    args = parser.parse_args()
    print(args)

    sns.set_theme(style='ticks', palette='Dark2')

    if args.participants == 'all':
        regex_filter = libemg.data_handler.RegexFilter('subject-', right_bound='/', values=[str(idx + 1).zfill(3) for idx in range(100)], description='')
        matching_directories = regex_filter.get_matching_files([path.as_posix() + '/' for path in Path('data').glob('*')])
        participants = [Path(participant).stem for participant in matching_directories]
        participants.sort()
    else:
        participants = str(args.participants).replace(' ', '').split(',')

    plotter = Plotter(participants, args.analysis, stage=args.stage)
    plotter.plot('fitts-metrics')
    plotter.plot('fitts-metrics-over-time')
    plotter.plot('fitts-traces')
    plotter.plot('heatmap')
    plotter.plot('loss')
    
    plt.show()
    print('-------------Analysis complete!-------------')


if __name__ == '__main__':
    main()

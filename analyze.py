from pathlib import Path
import math
import pickle
from argparse import ArgumentParser
import textwrap
from enum import Enum

import torch
import pandas as pd
import libemg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter, NullLocator
import seaborn as sns

from utils.adaptation import TIMEOUT, DWELL_TIME, Memory, ADAPTATION_TIME, VALIDATION_TIME
from utils.models import MLP
from experiment import MODELS, Participant, make_config, Config


RESULTS_DIRECTORY = 'results'


class Stage(Enum):
    ADAPTATION = 'adaptation'
    VALIDATION = 'validation'


class Plotter:
    def __init__(self, participants, dpi = 400, stage = None, presentation_layout = False):
        self.participants = participants
        self.dpi = dpi
        self.stage = stage
        self.presentation_layout = presentation_layout

        if self.stage == Stage.ADAPTATION:
            self.models = (MODELS[2], MODELS[3])  # reorder based on best visual for plots (oracle, ciil)
        else:
            self.models = MODELS

        self.stage_agnostic_results_path = Path(RESULTS_DIRECTORY)
        if len(participants) == 1:
            self.stage_agnostic_results_path = self.stage_agnostic_results_path.joinpath(self.participants[0].id)

        if self.stage is None:
            self.results_path = self.stage_agnostic_results_path
        else:
            self.results_path = self.stage_agnostic_results_path.joinpath(self.stage.value)
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.palette = sns.color_palette()
        self.model_palette = {format_names(model): color for model, color in zip(MODELS, self.palette)} # loop over all models to keep consistent colors for all plots

    def read_log(self, participant, model):
        config = make_config(participant, model)
        if self.stage == Stage.ADAPTATION:
            fitts_file = config.adaptation_fitts_file
        elif self.stage == Stage.VALIDATION:
            fitts_file = config.validation_fitts_file
        else:
            self._raise_stage_error('read Fitts log')

        return Log(fitts_file)

    def plot_throughput_over_time(self):
        if self.stage is None:
            self._raise_stage_error('plot throughput over time')

        data = {
            'Throughput (bits/s)': [],
            'Trials': [],
            'Model': []
        }
        fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
        t = np.arange(0, VALIDATION_TIME)
        for model in self.models:
            for participant in self.participants:
                log = self.read_log(participant, model)
                fitts_metrics = log.extract_fitts_metrics()
                throughput = moving_average(fitts_metrics['throughput'], window_size=10)
                data['Throughput (bits/s)'].extend(throughput)
                num_trials = len(throughput)
                data['Trials'].extend([idx + 1 for idx in range(num_trials)])
                data['Model'].extend([format_names(model) for _ in range(num_trials)])
                

        df = pd.DataFrame(data)
        sns.lineplot(df, x='Trials', y='Throughput (bits/s)', hue='Model', ax=ax, palette=self.model_palette)
        ax.set_title(f"Throughput Over {self.stage.value} Period".title())
        self._save_fig(fig, 'throughput-over-time.png')
        return fig

    def plot_fitts_metrics(self):
        if self.stage is None:
            self._raise_stage_error('plot Fitts metrics')

        metrics = {
            'Throughput (bits/s)': [],
            'Path Efficiency (%)': [],
            'Overshoots per Trial': [],
            '# Trials': [],
            'Completion Rate (%)': [],
            'Action Interference': [],
            'Drift': [],
            'Cost of Simultaneity': []
        }
        trial_info = {
            'Model': [],
            'Adaptive': [],
            'Subject ID': []
        }

        fig, axs = plt.subplots(nrows=2, ncols=4, layout='constrained', figsize=(12, 8), sharex=True)
        for model in self.models:
            for participant in self.participants:
                log = self.read_log(participant, model)
                config = make_config(participant, model)

                trial_info['Subject ID'].append(participant.id)
                trial_info['Model'].append(format_names(model))
                trial_info['Adaptive'].append('Yes' if config.model_is_adaptive else 'No')
                fitts_metrics = log.extract_fitts_metrics(exclude_warmup_trials=True)
                metrics['Throughput (bits/s)'].append(np.mean(fitts_metrics['throughput']))
                metrics['Path Efficiency (%)'].append(np.mean(fitts_metrics['efficiency']) * 100) # express as %
                metrics['Overshoots per Trial'].append(np.mean(fitts_metrics['overshoots']))
                metrics['# Trials'].append(fitts_metrics['num_trials'])
                metrics['Completion Rate (%)'].append(fitts_metrics['completion_rate'] * 100)   # express as %
                metrics['Action Interference'].append(np.nanmean(fitts_metrics['action_interference']))
                metrics['Drift'].append(np.nanmean(fitts_metrics['drift']))
                metrics['Cost of Simultaneity'].append(np.mean(fitts_metrics['cost_of_simultaneity']))

        data = {}
        data.update(metrics)
        data.update(trial_info)
        df = pd.DataFrame(data)
        df.to_csv(self.results_path.joinpath('fitts-metrics.csv'))

        axs = np.ravel(axs)   # flatten array - don't need it in a grid
        x = 'Model'
        hue = 'Adaptive'
        palette = {'Yes': self.palette[0], 'No': self.palette[1]} # want "yes" to be green (assumes Dark2 color palette)
        meanprops = {'markerfacecolor': 'black', 'markeredgecolor': 'black', 'marker': 'D'}
        formatted_model_names = format_names(self.models)

        try:
            stats_df = pd.read_csv(self.results_path.joinpath('stats.csv'))
            stats_df['p-value'] = pd.to_numeric(stats_df['p-value'], errors='coerce').fillna(0) # "p < 0.0001" converted to NaN, so we fill that with 0
        except FileNotFoundError:
            stats_df = None
            print('Stats file not found - not annotating boxplot.')

        for metric, ax in zip(metrics.keys(), axs):
            legend = 'auto' if ax == axs[3] else False # only plot legend on last axis
            if len(self.participants) == 1:
                sns.barplot(df, x=x, y=metric, ax=ax, hue=hue, legend=legend, palette=palette)
            else:
                sns.boxplot(df, x=x, y=metric, ax=ax, hue=hue, legend=legend, palette=palette, showmeans=True, meanprops=meanprops) # maybe color boxes based on intended and unintended RMSE? or experience level? or have three box plots: within, combined, and all?
            
            if stats_df is not None:
                bottom, top = ax.get_ylim()
                y_axis_range = top - bottom
                level = 0
                for _, row in stats_df.iterrows():
                    stats_metric, comparison, p = row
                    if stats_metric not in metric:
                        # Metric names in stats don't always line up with dictionary keys
                        continue

                    if p < 0.001:
                        symbol = '***'
                    elif p < 0.01:
                        symbol = '**'
                    elif p < 0.05:
                        symbol = '*'
                    else:
                        # Only want to plot bars for significant differences
                        continue

                    comparison = comparison.replace('(', '').replace(')', '')
                    compared_models = [formatted_model_names.index(compared_model) for compared_model in comparison.split(' - ')]
                    x1, x2 = compared_models
                    bar_height = (y_axis_range * 0.07 * level) + top
                    bar_tips = bar_height - (y_axis_range * 0.02)
                    ax.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], linewidth=1, c='k')
                    ax.text((x1 + x2) * 0.5, bar_height - (y_axis_range * 0.02), symbol, ha='center', va='bottom', c='k')
                    level += 1

            if '%' in metric:
                percent_ticks = [tick for tick in ax.get_yticks() if tick <= 100 and tick >= 0]
                ax.set_yticks(percent_ticks)


        symbol_handles = [
            mlines.Line2D([], [], color=meanprops['markerfacecolor'], marker=meanprops['marker'], linewidth=0, label='Mean'),
            mlines.Line2D([], [], markerfacecolor='white', markeredgecolor='black', marker='o', linewidth=0, label='Outlier'),
        ]
        fig.legend(handles=symbol_handles, title='Symbols', loc='outside upper left', ncols=len(symbol_handles))
        color_legend = axs[3].get_legend()
        fig.legend(handles=color_legend.legend_handles, title='Adaptive', loc='outside upper right', ncols=len(color_legend.legend_handles))
        color_legend.remove()

        fig.suptitle('Online Usability Metrics')
        self._save_fig(fig, 'fitts-metrics.png')
        return fig

    def plot_fitts_traces(self):
        if self.stage is None:
            self._raise_stage_error('plot Fitts traces')

        fig, axs = plt.subplots(nrows=1, ncols=len(self.models), figsize=(14, 8), layout='constrained', sharex=True, sharey=True)
        cmap = mpl.colormaps['Dark2']
        lines = []
        for model, ax in zip(self.models, axs):
            for participant_idx, participant in enumerate(self.participants):
                log = self.read_log(participant, model)
                traces = [trial.cursor_positions for trial in log.trials]
                for trace in traces:
                    lines.extend(ax.plot(trace[:, 0], trace[:, 1], c=cmap(participant_idx), label=participant.id, linewidth=1, alpha=0.5))
                
            ax.set_title(format_names(model))
            ax.set_xlabel('X (Pixels)')
        
        axs[0].set_ylabel('Y (Pixels)')
        if len(self.participants) != 1:
            legend_handles = get_unique_legend_handles(lines)
            axs[-1].legend(legend_handles.values(), format_names(legend_handles.keys()))

        fig.suptitle('Fitts Traces')
        self._save_fig(fig, 'fitts-traces.png')
        return fig

    def plot_dof_activation_heatmap(self):
        if self.stage is None:
            self._raise_stage_error('plot heatmap')

        # Create heatmap where x is DOF 1 and y is DOF2
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        outer_grid = fig.add_gridspec(nrows=2, ncols=2)
        width_ratios = [2, 1]
        height_ratios = [1, 2]
        bins = np.round(np.arange(-1.1, 1.2, 0.2), 2)  # extend past 1 to include 1 in arange
        bin_ticks = np.arange(bins.shape[0])
        hist_ticks = np.array([1e2, 1e3, 1e4, 1e5])
        bbox = {'boxstyle': 'round'}
        x_hist_axs = []
        y_hist_axs = []
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
            x_hist_axs.append(x_hist_ax)
            y_hist_axs.append(y_hist_ax)

            # Plot
            # nm_mask = np.all(np.abs(model_predictions) < bins[bins.shape[0] // 2], axis=1)
            # model_predictions = model_predictions[~nm_mask]
            simultaneous_mask = np.all(np.abs(model_predictions) > 1e-3, axis=1)    # predictions are never exactly 0
            simultaneity = np.sum(simultaneous_mask) / model_predictions.shape[0]
            x_predictions = model_predictions[:, 0]
            y_predictions = model_predictions[:, 1]
            x_y_counts, _, _ = np.histogram2d(x_predictions, y_predictions, bins=bins, density=False)   # density sets it to return pdf, not % occurrences
            x_hist_ax.hist(x_predictions, bins=bins)
            y_hist_ax.hist(y_predictions, bins=bins, orientation='horizontal')
            # Flip heatmap y bins so they align with 1D histograms and show bins in ascending order
            heatmap = sns.heatmap(np.flip(x_y_counts, axis=0), ax=heatmap_ax, cmap=sns.light_palette('seagreen', as_cmap=True), norm=mpl.colors.LogNorm())

            # Formatting
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.yaxis.set_minor_locator(NullLocator())  # disable minor (logarithmic) ticks
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=x_predictions.shape[0], decimals=1))
            heatmap_ax.set_xticks(bin_ticks, bins, rotation=90)
            heatmap_ax.set_yticks(bin_ticks, np.flip(bins), rotation=0)
            heatmap_ax.set_xlabel('Open / Close Activation')
            heatmap_ax.set_ylabel('Pro / Supination Activation')
            x_hist_ax.set_xticks([])    # in line with heatmap, so ticks aren't needed
            x_hist_ax.set_xlim(bins[0], bins[-1])
            y_hist_ax.set_yticks([])    # in line with heatmap, so ticks aren't needed
            y_hist_ax.set_ylim(bins[0], bins[-1])
            x_hist_ax.set_title(format_names(model))
            x_hist_ax.set_ylabel('Frequency')
            y_hist_ax.set_xlabel('Frequency')
            x_hist_ax.set_yscale('log')
            x_hist_ax.set_yticks(hist_ticks)
            x_hist_ax.minorticks_off()
            y_hist_ax.set_xscale('log')
            y_hist_ax.set_xticks(hist_ticks)
            y_hist_ax.minorticks_off()
            text_x = y_hist_ax.get_position().x0
            text_y = x_hist_ax.get_position().y0
            fig.text(text_x, text_y, f"Simultaneity: {100 * simultaneity:.1f}%",
                     ha='center', va='bottom', fontsize=12, fontweight='bold', bbox=bbox)
            
        # Need to go through and align all histogram axes with eachother for consistent dimensions across subgrids
        for x_hist_ax, y_hist_ax in zip(x_hist_axs, y_hist_axs):
            x_hist_ax.sharey(x_hist_axs[0])
            y_hist_ax.sharex(y_hist_axs[0])

        fig.suptitle('Activation Heatmap')
        self._save_fig(fig, 'heatmap.png')
        return fig

    def plot_loss(self):
        fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
        epochs = []
        losses = []
        model_labels = []
        for model in self.models:
            for participant in self.participants:
                config = make_config(participant, model)
                loss_df = pd.read_csv(config.loss_file, header=None)
                batch_timestamps = loss_df.iloc[:, 0]
                batch_losses = loss_df.iloc[:, 1]
                epoch_timestamps = np.unique(batch_timestamps)
                assert (len(epoch_timestamps) == config.NUM_TRAIN_EPOCHS) or config.model_is_adaptive, f"Unexpected number of epochs in loss file. Expected {config.NUM_TRAIN_EPOCHS}, but got {len(epoch_timestamps)} for {participant.id} {model}."
                epoch_losses = [np.mean(batch_losses[batch_timestamps == timestamp]) for timestamp in epoch_timestamps]
                epochs.extend(list(range(len(epoch_timestamps))))
                losses.extend(epoch_losses)
                model_labels.extend(format_names([model for _ in range(len(epoch_timestamps))]))

        df = pd.DataFrame({
            'Epochs': epochs,
            'L1 Loss': losses,
            'Model': model_labels
        })
        sns.lineplot(df, x='Epochs', y='L1 Loss', hue='Model', ax=ax, palette=self.model_palette)
        ax.axvline(Config.NUM_TRAIN_EPOCHS, color='black', linestyle='--', label='SGT Epochs')
        ax.annotate('SGT Training Epochs', 
             xy=(Config.NUM_TRAIN_EPOCHS, ax.get_yticks()[-2]),
             xytext=(Config.NUM_TRAIN_EPOCHS + 100, ax.get_yticks()[-2]),  # offset for better placement
             arrowprops=dict(arrowstyle='->', color='black'),
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
             fontsize=10)
        ax.set_title('Training Loss')
        self._save_fig(fig, 'loss.png', stage_agnostic=True)
        return fig

    def plot_num_reps_wsgt(self):
        fig, ax = plt.subplots()
        rep_ranges = [idx for idx in range(1, 7)]
        metrics = {
            'MAE': [],
            '# reps': [],
        }
        fe = libemg.feature_extractor.FeatureExtractor()
        for participant in self.participants:
            test_config = make_config(participant, 'combined-sgt')
            test_odh = test_config.load_sgt_data().isolate_data('reps', [0])   # just grab first rep
            test_windows, test_metadata = test_odh.parse_windows(test_config.window_length, test_config.window_increment, metadata_operations={'labels': 'last_sample'})
            test_labels = test_metadata['labels']
            test_features = fe.extract_features(test_config.features, test_windows, test_config.feature_dictionary, array=True)

            for num_reps in rep_ranges:
                train_config = make_config(participant, 'within-sgt')
                train_odh = train_config.load_sgt_data().isolate_data('reps', [idx for idx in range(num_reps)])
                assert len(train_odh.data) == num_reps, f"Expected {num_reps} files, but got {len(train_odh.data)}."
                train_windows, train_metadata = train_odh.parse_windows(train_config.window_length, train_config.window_increment, metadata_operations={'labels': 'last_sample'})
                train_labels = train_metadata['labels']
                train_features = fe.extract_features(train_config.features, train_windows, train_config.feature_dictionary, array=True)

                mdl = MLP(train_features.shape[1], Config.BATCH_SIZE, Config.LEARNING_RATE, Config.LOSS_FUNCTION)
                memory = Memory()
                memory.add_memories(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
                mdl.fit(memory, Config.NUM_TRAIN_EPOCHS)

                predictions = mdl.predict(test_features)
                mae = np.abs(predictions - test_labels.astype(predictions.dtype)).mean(axis=0).mean()
                metrics['MAE'].append(mae)
                metrics['# reps'].append(num_reps)

        df = pd.DataFrame(metrics)
        sns.boxplot(df, x='# reps', y='MAE', ax=ax)
        fig.suptitle('Impact of # Reps During Training for W-SGT')
        self._save_fig(fig, 'offline.png', stage_agnostic=True)
        return fig

    def plot_prompt_labels(self):
        config = make_config(self.participants[0], 'within-sgt')
        odh = config.load_sgt_data()
        labels = odh.labels[0]
        fig, ax = plt.subplots(layout='constrained')
        t = np.linspace(0, 50, num=labels.shape[0]) # 1 rep was 50 seconds
        ax.plot(t, labels[:, 0], label='Hand Open/Close')
        ax.plot(t, labels[:, 1], label='Wrist Rotation')
        ax.set_title('Prompt Trajectory')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('DoF Activation')
        ax.legend(ncols=2, loc='upper right')
        ax.set_ylim((-1.25, 1.25))
        self._save_fig(fig, 'prompt.png')
        return fig

    def plot_decision_stream(self):
        if self.stage is None:
            self._raise_stage_error('plot decision stream')

        fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
        log = self.read_log(self.participants[12], 'ciil')

        decision_stream_predictions = []
        decision_stream_timestamps = []
        for trial in log.trials:
            decision_stream_predictions.extend(trial.predictions)
            decision_stream_timestamps.extend(trial.prediction_timestamps)
        decision_stream_predictions = np.array(decision_stream_predictions)

        if self.stage == Stage.ADAPTATION:
            memory = Memory()
            memory.from_file(Path(log.path).parent.as_posix(), 1000)
            pseudolabels = []
            ignored = []
            for timestamp in decision_stream_timestamps:
                try:
                    idx = memory.experience_timestamps.index(timestamp)
                    pseudolabels.append([timestamp, memory.experience_targets[idx, 1].item()])
                except ValueError:
                    ignored.append([timestamp, 0])

            pseudolabels = np.array(pseudolabels)
            ignored = np.array(ignored)
            ax.scatter(pseudolabels[:, 0], pseudolabels[:, 1], s=2, label='Pseudolabels')
            ax.scatter(ignored[:, 0], ignored[:, 1], s=2, label='Ignored')

        else:
            ax.plot(decision_stream_timestamps, decision_stream_predictions[:, 1])

        ax.set_xlabel('Timestamps')
        ax.set_ylabel('Pronation / Supination Activation')
        ax.set_title(f"Decision Stream ({self.stage.value})".title())
        self._save_fig(fig, 'decision-stream.png')

    def plot_survey_results(self):
        if len(self.participants) == 1:
            print('Only 1 participant found - skipping plot of survey results.')
            return
        df = pd.read_csv(self.stage_agnostic_results_path.joinpath('intra-survey-results.csv'))
        questions = [ 
            'I had good control of isolated motions (e.g., move right)',
            'I had good control of combined motions (e.g., move up and right)',
            'I could stop and stay at rest when I wanted to',
            'I felt in control of the cursor',
            'I could control the speed well',
            'The training phase was engaging'
        ]
        responses = {
            'Strongly agree': '#00ff00',
            'Agree': '#008800',
            'Neutral': '#000000',
            'Disagree': '#880000',
            'Strongly disagree': '#ff0000',
        }

        columns = list(responses.keys()) + ['Model', 'Question']
        fig, axs = plt.subplots(nrows=3, ncols=2, layout='constrained', figsize=(8, 6), sharex=True, sharey=True)
        for question, ax in zip(questions, np.ravel(axs)):
            data = {column: [] for column in columns}

            for model in MODELS:
                model_mask = df['Condition'] == format_names(model)
                data['Model'].append(format_names(model))
                data['Question'].append(question)
                counts = df[model_mask][question].value_counts()
                for response in responses.keys():
                    if response not in counts:
                        count = 0
                    else:
                        count = counts[response]
                    data[response].append(count)

            question_df = pd.DataFrame(data)
            question_df.plot.barh(x='Model', stacked=True, ax=ax, legend=False, color=responses)
            ax.set_title(textwrap.fill(question, width=40))

        axs[0, 0].set_xticks(np.arange(0, axs[0, 0].get_xlim()[1]))
        axs[0, 0].minorticks_off()
        xlabel = 'Responses'
        axs[-1, 0].set_xlabel(xlabel)
        axs[-1, 1].set_xlabel(xlabel)
        fig.suptitle('Survey Results')
        handles = [mpatches.Patch(color=color, label=label) for label, color in responses.items()]
        fig.legend(handles=handles, ncols=len(responses), loc='outside lower center')
        self._save_fig(fig, 'survey.png', stage_agnostic=True)
        return fig

    def _save_fig(self, fig, filename, stage_agnostic = False):
        if stage_agnostic:
            results_path = self.stage_agnostic_results_path
        else:
            results_path = self.results_path
        filepath = results_path.joinpath(filename)
        fig.savefig(filepath, dpi=self.dpi)
        print(f"File saved to {filepath.as_posix()}.")

    def _raise_stage_error(self, task):
        raise ValueError(f"Stage must be specified when trying to perform task: {task}. Current stage value: {self.stage}.")


class Log:
    def __init__(self, path, analysis = None) -> None:
        self.path = path
        self.analysis = analysis
        if self.analysis == 'within':
            self.exclude_within_dof_trials = False
            self.exclude_combined_dof_trials = True
        elif self.analysis == 'combined':
            self.exclude_within_dof_trials = True
            self.exclude_combined_dof_trials = False
        elif analysis is None:
            self.exclude_within_dof_trials = False
            self.exclude_combined_dof_trials = False
        else:
            raise ValueError(f"Unexpected value for analysis. Got: {self.analysis}.")

        self.exclude_timeout_trials = False

        with open(self.path, 'rb') as f:
            run_log = pickle.load(f)

        # Using trial mask to get relevant samples - convert to arrays for convenience
        for key in run_log.keys():
            run_log[key] = np.array(run_log[key])

        all_trials = [Trial(run_log, idx) for idx in np.unique(run_log['trial_number'])[:-1]] # skip the final trial b/c it is always incomplete

        trials = []
        for t in all_trials:
            if t.is_timeout_trial and self.exclude_timeout_trials:
                print(f"Skipping trial {t.trial_idx} in log {self.path}.")
                continue

            trials.append(t)
            
        self.trials = trials

    def extract_fitts_metrics(self, exclude_warmup_trials = False):
        trials = self.trials
        if exclude_warmup_trials:
            num_warmup_trials = 20  # based on average plot of throughput over time
            trials = trials[num_warmup_trials:]

        fitts_results = {
            'timeouts': [],
            'overshoots': [],
            'efficiency': [],
            'throughput': [],
            'num_trials': -1,
            'completion_rate': -1,
            'action_interference': [],
            'drift': [],
            'time': [],
            'cost_of_simultaneity': []
        }
        
        total_time = 0
        for t in trials:
            if t.is_timeout_trial:
                fitts_results['timeouts'].append(t.trial_idx)

            fitts_results['throughput'].append(t.calculate_throughput())
            fitts_results['overshoots'].append(t.calculate_overshoots())
            fitts_results['efficiency'].append(t.calculate_efficiency())
            fitts_results['action_interference'].append(t.calculate_action_interference())
            fitts_results['drift'].append(t.calculate_drift())
            fitts_results['cost_of_simultaneity'].append(t.calculate_cost_of_simultaneity())

            total_time += t.trial_time
            fitts_results['time'].append(total_time)

        fitts_results['num_trials'] = len(trials)
        fitts_results['completion_rate'] = 1 - (len(fitts_results['timeouts']) / fitts_results['num_trials'])
            
        return fitts_results


class Trial:
    def __init__(self, run_log, trial_idx):
        self.trial_idx = trial_idx

        trial_mask = np.where(run_log['trial_number'] == self.trial_idx)[0]
        cursor_positions = run_log['cursor_position'][trial_mask]
        targets = run_log['goal_target'][trial_mask]
        assert np.all(targets == targets[0]), f"Found trial with multiple targets."
        assert np.all(cursor_positions[:, 2] == cursor_positions[0, 2]), f"Found trial with multiple cursor widths."
        self.cursor_width = cursor_positions[0, 2]
        self.cursor_positions = cursor_positions[:, :2]
        target = targets[0]
        self.target_position = target[:2]
        self.target_width = target[2]
        self.clock_timestamps = run_log['global_clock'][trial_mask]
        self.prediction_timestamps = run_log['time_stamp'][trial_mask]
        model_output = run_log['class_label'][trial_mask]
        self.predictions = np.array([list(map(float, model_output.replace('[', '').replace(']', '').split(','))) for model_output in model_output])
        self.trial_time = self.clock_timestamps[-1] - self.clock_timestamps[0]
        self.is_timeout_trial = self.trial_time >= (TIMEOUT * 0.98)   # account for rounding errors
    
    def in_target(self, cursor):
        return math.dist(cursor, self.target_position) < (self.target_width // 2 + self.cursor_width // 2)

    def calculate_efficiency(self):
        distance_travelled = np.sum([math.dist(self.cursor_positions[i], self.cursor_positions[i-1]) for i in range(1, len(self.cursor_positions))])
        fastest_path = math.dist(self.cursor_positions[0], self.target_position)    # TODO: Should this be based on the cursor position or where the previous target was?
        return fastest_path / distance_travelled

    def calculate_throughput(self):
        trial_time = self.trial_time
        if not self.is_timeout_trial:
            # Only subtract dwell time for successful trials
            trial_time -= DWELL_TIME
        starting_cursor_position = self.cursor_positions[0]
        distance = math.dist(starting_cursor_position, self.target_position)
        id = math.log2(distance / self.target_width + 1)
        return id / trial_time

    def calculate_overshoots(self):
        in_bounds = [self.in_target(self.cursor_positions[i]) for i in range(0, len(self.cursor_positions))]
        overshoots = 0
        for i in range(1,len(in_bounds)):
            if in_bounds[i-1] == True and in_bounds[i] == False:
                overshoots += 1 
        return overshoots

    def calculate_action_interference(self):
        action_interference_predictions = []
        for cursor_position, prediction in zip(self.cursor_positions, self.predictions):
            in_line_with_target_mask = np.where(np.abs(self.target_position - cursor_position) <= (self.target_width // 2 + self.cursor_width // 2))[0]
            if len(in_line_with_target_mask) != 1:
                # In target or not in line in either component, so we skip this
                continue
            wrong_component_magnitude = np.abs(prediction[in_line_with_target_mask[0]])
            action_interference_predictions.append(wrong_component_magnitude)
        if len(action_interference_predictions) == 0:
            # Can't determine action interference for this trial - exclude it
            return np.nan
        return np.mean(action_interference_predictions)

    def calculate_drift(self):
        drift_predictions = []
        for cursor_position, prediction in zip(self.cursor_positions, self.predictions):
            if not self.in_target(cursor_position):
                continue
            prediction_magnitude = np.sum(np.abs(prediction))
            drift_predictions.append(prediction_magnitude)
        if len(drift_predictions) == 0:
            # Can't determine drift for this trial - exclude it
            return np.nan
        return np.mean(drift_predictions)

    def calculate_cost_of_simultaneity(self):
        simultaneous_mask = np.all(np.abs(self.predictions) > 1e-3, axis=1)    # predictions are never exactly 0
        simultaneity = np.sum(simultaneous_mask) / self.predictions.shape[0]
        # return 1 - (simultaneity * self.calculate_efficiency() * (1 - self.calculate_action_interference()))
        # return self.calculate_throughput() * (self.calculate_action_interference() + self.calculate_drift() + (1 - self.calculate_efficiency()))
        # return self.calculate_efficiency() / simultaneity
        # return simultaneity / (self.calculate_efficiency() * self.calculate_throughput())
        return simultaneity / self.calculate_efficiency()


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
            ('Subject-', 'S'),
            ('Uc', 'UC')
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


def calculate_participant_metrics(participants):
    ages = []
    males = 0
    females = 0
    for participant in participants:
        ages.append(participant.age)
        if participant.sex == 'M':
            males += 1
        elif participant.sex == 'F':
            females += 1
        else:
            raise ValueError(f"Unexpected value for participant.sex. Got: {participant.sex}.")
    
    print(f"Age: {min(ages)}-{max(ages)}\tM: {males}\tF: {females}")


def main():
    parser = ArgumentParser(prog='Analyze offline data.')
    parser.add_argument('-p', '--participants', default='all', help='List of participants to evaluate.')
    args = parser.parse_args()
    print(args)

    sns.set_theme(style='ticks', palette='Dark2')

    if args.participants == 'all':
        regex_filter = libemg.data_handler.RegexFilter('subject-', right_bound='/', values=[str(idx + 1).zfill(3) for idx in range(100)], description='')
        matching_directories = regex_filter.get_matching_files([path.as_posix() + '/' for path in Path('data').glob('*')])
        participant_ids = [Path(participant).stem for participant in matching_directories]
        participant_ids.sort()
    else:
        participant_ids = str(args.participants).replace(' ', '').split(',')


    participants = []
    for participant_id in participant_ids:
        participant_files = [file for file in Path('data').rglob('participant.json') if participant_id in file.as_posix() and 'archive' not in file.as_posix()]
        assert len(participant_files) == 1, f"Expected a single matching participant file for {participant_id}, but got {participant_files}."
        participants.append(Participant.load(participant_files[0]))

    calculate_participant_metrics(participants)
    # TODO: Maybe look at pulling apart performance for novice vs. more experienced users
    # TODO: Add 'paper' vs. 'presentation' parameters to Plotter to adjust figure layouts
    validation_plotter = Plotter(participants, stage=Stage.VALIDATION)
    validation_plotter.plot_fitts_metrics()
    validation_plotter.plot_throughput_over_time()
    validation_plotter.plot_dof_activation_heatmap()
    validation_plotter.plot_loss()
    validation_plotter.plot_survey_results()

    adaptation_plotter = Plotter(participants, stage=Stage.ADAPTATION)
    adaptation_plotter.plot_throughput_over_time()
    
    plt.show()
    print('-------------Analysis complete!-------------')


if __name__ == '__main__':
    main()

from pathlib import Path
import math
import pickle
from argparse import ArgumentParser
import textwrap

import pygame
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

from utils.adaptation import TIMEOUT, DWELL_TIME, Memory, ADAPTATION_TIME, VALIDATION_TIME, TARGET_RADIUS, ISOFITTS_RADIUS
from utils.models import MLP
from experiment import MODELS, Participant, make_config, Config


RESULTS_DIRECTORY = 'results'
ADAPTATION = 'adaptation'
VALIDATION = 'validation'
REPORT = 'report'
PRESENTATION = 'presentation'
THESIS = 'thesis'


class Plotter:
    def __init__(self, participants, dpi = 400, layout = REPORT):
        self.participants = participants
        self.dpi = dpi
        self.layout = layout
        self.validation_models = MODELS
        self.adaptation_models = (MODELS[2], MODELS[3])

        self.results_path = Path(RESULTS_DIRECTORY)
        if len(participants) == 1:
            self.results_path = self.results_path.joinpath(self.participants[0].id)

        self.results_path.mkdir(parents=True, exist_ok=True)

        self.palette = sns.color_palette()
        self.model_palette = {format_names(model): color for model, color in zip(MODELS, self.palette)} # loop over all models to keep consistent colors for all plots

    def read_log(self, participant, model, stage):
        config = make_config(participant, model)
        if stage == ADAPTATION:
            fitts_file = config.adaptation_fitts_file
        elif stage == VALIDATION:
            fitts_file = config.validation_fitts_file
        else:
            self._raise_stage_error(stage)

        return Log(fitts_file)

    def plot_throughput_over_time(self):
        if self.layout == THESIS:
            fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True, layout='constrained', figsize=(6, 8))
            loc = 'upper center'
            bbox_to_anchor = (0.5, -0.15)
            ncols = len(self.validation_models)
        elif self.layout == PRESENTATION:
            fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True, layout='constrained', figsize=(6, 4))
            loc = 'upper center'
            bbox_to_anchor = None
            ncols = None
        else:
            fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, layout='constrained', figsize=(12, 4))
            loc = 'center left'
            bbox_to_anchor = (1, 0.5)
            ncols = 1
        self.plot_throughput_over_stage(ADAPTATION, ax=axs[0])
        self.plot_throughput_over_stage(VALIDATION, ax=axs[1])

        if self.layout in (THESIS, PRESENTATION):
            axs[0].set_xlabel('')   # don't want to duplicate xlabel
        axs[0].get_legend().remove()    # don't want duplicate legend
        
        if self.layout == PRESENTATION:
            axs[1].get_legend().remove()    # don't want legend for thesis presentation
        else:
            sns.move_legend(axs[1], loc=loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols)

        self._save_fig(fig, 'throughput-over-time.png')
        return fig

    def plot_throughput_over_stage(self, stage, ax = None):
        if stage == ADAPTATION:
            stage_time = ADAPTATION_TIME
            legend_loc = 'upper left'
            models = self.adaptation_models
        elif stage == VALIDATION:
            stage_time = VALIDATION_TIME
            legend_loc = 'upper right'
            models = self.validation_models
        else:
            self._raise_stage_error(stage)

        if ax is None:
            fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
        else:
            fig = None

        data = {
            'Throughput (bits/s)': [],
            'Model': [],
            'Time (seconds)': []
        }
        t = np.arange(0, stage_time)
        for model in models:
            for participant in self.participants:
                log = self.read_log(participant, model, stage)
                fitts_metrics = log.extract_fitts_metrics()
                throughput = np.array([0] + fitts_metrics['throughput'])    # start with a tp of 0
                timestamps = np.array([0] + fitts_metrics['time'])  # start with a timestamp of 0
                align_mask = np.searchsorted(timestamps, t, side='right') - 1    # hold throughput until a new trial is completed
                throughput = throughput[align_mask]
                throughput = moving_average(throughput, window_size=60)
                data['Throughput (bits/s)'].extend(throughput)
                data['Time (seconds)'].extend(np.linspace(0, t[-1], num=throughput.shape[0]))
                data['Model'].extend([format_names(model) for _ in range(throughput.shape[0])])
                
        df = pd.DataFrame(data)
        sns.lineplot(df, x='Time (seconds)', y='Throughput (bits/s)', hue='Model', ax=ax, palette=self.model_palette)
        ax.set_title(f"Throughput Over {stage} Period".title())
        sns.move_legend(ax, loc=legend_loc, ncols=len(models))
        if fig is not None:
            self._save_fig(fig, f"throughput-over-{stage}.png")
        return fig

    def plot_fitts_metrics(self, stage):
        if stage == ADAPTATION:
            models = self.adaptation_models
        elif stage == VALIDATION:
            models = self.validation_models
        else:
            self._raise_stage_error(stage)

        metrics = {
            'Throughput (bits/s)': [],
            'Path Efficiency (%)': [],
            'Overshoots per Trial': [],
            '# Trials': [],
            'Completion Rate (%)': [],
            'Action Interference': [],
            'Drift': [],
            'Simultaneity Gain': []
        }
        trial_info = {
            'Model': [],
            'Adaptive': [],
            'Subject ID': []
        }

        if self.layout == THESIS:
            figsize = (8, 8)
            xtick_rotation = 90
            bar_tip_multiplier = 0.02
            bar_text_multiplier = 0.02
        elif self.layout == PRESENTATION:
            figsize = (12.5, 5)
            xtick_rotation = 0
            bar_tip_multiplier = 0.01
            bar_text_multiplier = 0.06
        else:
            figsize = (12, 8)
            xtick_rotation = 0
            bar_tip_multiplier = 0.02
            bar_text_multiplier = 0.02

        fig, axs = plt.subplots(nrows=2, ncols=4, layout='constrained', figsize=figsize, sharex=True)
        for model in models:
            for participant in self.participants:
                log = self.read_log(participant, model, stage)
                config = make_config(participant, model)

                trial_info['Subject ID'].append(participant.id)
                trial_info['Model'].append(format_names(model))
                trial_info['Adaptive'].append('Yes' if config.model_is_adaptive else 'No')
                fitts_metrics = log.extract_fitts_metrics(exclude_learning_trials=True)
                metrics['Throughput (bits/s)'].append(np.mean(fitts_metrics['throughput']))
                metrics['Path Efficiency (%)'].append(np.mean(fitts_metrics['efficiency']) * 100) # express as %
                metrics['Overshoots per Trial'].append(np.mean(fitts_metrics['overshoots']))
                metrics['# Trials'].append(fitts_metrics['num_trials'])
                metrics['Completion Rate (%)'].append(fitts_metrics['completion_rate'] * 100)   # express as %
                metrics['Action Interference'].append(np.nanmean(fitts_metrics['action_interference']))
                metrics['Drift'].append(np.nanmean(fitts_metrics['drift']))
                metrics['Simultaneity Gain'].append(np.nanmean(fitts_metrics['simultaneity_gain']))

        data = {}
        data.update(metrics)
        data.update(trial_info)
        df = pd.DataFrame(data)
        df.to_csv(self.results_path.joinpath(f"{stage}-fitts-metrics.csv"))

        axs = np.ravel(axs)   # flatten array - don't need it in a grid
        x = 'Model'
        hue = 'Adaptive'
        palette = {'Yes': self.palette[0], 'No': self.palette[1]} # want "yes" to be green (assumes Dark2 color palette)
        meanprops = {'markerfacecolor': 'black', 'markeredgecolor': 'black', 'marker': 'D'}
        formatted_model_names = format_names(models)

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
                    bar_tips = bar_height - (y_axis_range * bar_tip_multiplier)
                    ax.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], linewidth=1, c='k')
                    ax.text((x1 + x2) * 0.5, bar_height - (y_axis_range * bar_text_multiplier), symbol, ha='center', va='bottom', c='k')
                    level += 1

            if '%' in metric:
                percent_ticks = [tick for tick in ax.get_yticks() if tick <= 100 and tick >= 0]
                ax.set_yticks(percent_ticks)

            ax.tick_params(axis='x', rotation=xtick_rotation)

        symbol_handles = [
            mlines.Line2D([], [], color=meanprops['markerfacecolor'], marker=meanprops['marker'], linewidth=0, label='Mean'),
            mlines.Line2D([], [], markerfacecolor='white', markeredgecolor='black', marker='o', linewidth=0, label='Outlier'),
        ]
        color_legend = axs[3].get_legend()

        if self.layout == PRESENTATION:
            axs[3].legend(handles=color_legend.legend_handles, title='Adaptive', loc='upper left', bbox_to_anchor=(1, 1))
            axs[7].legend(handles=symbol_handles, title='Symbols', loc='lower left', bbox_to_anchor=(1, 0))
        else:
            fig.legend(handles=symbol_handles, title='Symbols', loc='outside upper left', ncols=len(symbol_handles))
            fig.legend(handles=color_legend.legend_handles, title='Adaptive', loc='outside upper right', ncols=len(color_legend.legend_handles))
            color_legend.remove()

        fig.suptitle('Online Usability Metrics')
        self._save_fig(fig, f"{stage}-fitts-metrics.png")
        return fig

    def plot_fitts_traces(self, stage):
        if stage == ADAPTATION:
            models = self.adaptation_models
        elif stage == VALIDATION:
            models = self.validation_models
        else:
            self._raise_stage_error(stage)

        fig, axs = plt.subplots(nrows=1, ncols=len(models), figsize=(14, 8), layout='constrained', sharex=True, sharey=True)
        cmap = mpl.colormaps['Dark2']
        lines = []
        for model, ax in zip(models, axs):
            for participant_idx, participant in enumerate(self.participants):
                log = self.read_log(participant, model, stage)
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
        self._save_fig(fig, f"{stage}-fitts-traces.png")
        return fig

    def plot_dof_activation_heatmap(self, stage, show_histograms = False):
        if stage == ADAPTATION:
            models = self.adaptation_models
        elif stage == VALIDATION:
            models = self.validation_models
        else:
            self._raise_stage_error(stage)

        if show_histograms:
            fig = plt.figure(figsize=(10, 10), constrained_layout=True)
            axs = None
            outer_grid = fig.add_gridspec(nrows=2, ncols=2)
            axis_cbar = True
        elif self.layout == THESIS:
            fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained', figsize=(6, 6.5))
            axs = np.ravel(axs)
            outer_grid = None
            axis_cbar = False
        else:
            fig, axs = plt.subplots(nrows=1, ncols=4, layout='constrained', figsize=(12, 3))
            outer_grid = None
            axis_cbar = True

        width_ratios = [2, 1]
        height_ratios = [1, 2]
        bins = np.round(np.arange(-1.1, 1.2, 0.2), 2)  # extend past 1 to include 1 in arange
        bin_ticks = np.arange(bins.shape[0])
        hist_ticks = np.array([1e2, 1e3, 1e4, 1e5])
        bbox = {'boxstyle': 'round'}
        x_hist_axs = []
        y_hist_axs = []
        predictions = {}
        vmin = None
        vmax = None
        for model in models:
            model_predictions = []
            for participant in self.participants:
                log = self.read_log(participant, model, stage)
                log_predictions = [trial.predictions for trial in log.trials]
                log_predictions = np.concatenate(log_predictions)
                model_predictions.append(log_predictions)
            model_predictions = np.concatenate(model_predictions)
            x_y_counts, _, _ = np.histogram2d(model_predictions[:, 0], model_predictions[:, 1], bins=bins, density=False)   # density sets it to return pdf, not % occurrences
            vmin = x_y_counts.min() if vmin is None else min(vmin, x_y_counts.min())
            vmax = x_y_counts.max() if vmax is None else max(vmax, x_y_counts.max())
            predictions[model] = model_predictions

        assert vmin is not None and vmax is not None, 'Error calculating min and max values for colorbar normalization.'
        vmin = max(vmin, 1) # can't have a log value of 0
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        for model_idx, model in enumerate(models):
            model_predictions = predictions[model]
            simultaneous_mask = np.all(np.abs(model_predictions) > 1e-3, axis=1)    # predictions are never exactly 0
            simultaneity = np.sum(simultaneous_mask) / model_predictions.shape[0]
            x_predictions = model_predictions[:, 0]
            y_predictions = model_predictions[:, 1]
            x_y_counts, _, _ = np.histogram2d(x_predictions, y_predictions, bins=bins, density=False)   # density sets it to return pdf, not % occurrences

            if outer_grid is not None:
                # Format heatmap + histogram axes
                inner_grid = outer_grid[model_idx].subgridspec(nrows=2, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios)
                axs = inner_grid.subplots()
                heatmap_ax = axs[1, 0]
                x_hist_ax = axs[0, 0]
                y_hist_ax = axs[1, 1]
                axs[0, 1].set_axis_off()    # hide unused axis
                cbar = True
            else:
                assert axs is not None, 'axs not defined.'
                heatmap_ax = axs[model_idx]
                cbar = model == models[-1] and axis_cbar
                x_hist_ax = None
                y_hist_ax = None


            # Flip heatmap y bins so they align with 1D histograms and show bins in ascending order
            heatmap = sns.heatmap(np.flip(x_y_counts, axis=0), ax=heatmap_ax,
                                  cmap=sns.light_palette('seagreen', as_cmap=True), norm=norm, cbar=cbar)

            # Formatting
            if cbar:
                colorbar = heatmap.collections[0].colorbar
                colorbar.ax.yaxis.set_minor_locator(NullLocator())  # disable minor (logarithmic) ticks
                colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=x_predictions.shape[0], decimals=1))
            elif not axis_cbar and model == models[-1]:
                colorbar = fig.colorbar(heatmap.collections[0], ax=axs[2:], orientation='horizontal')
                colorbar.ax.xaxis.set_minor_locator(NullLocator())  # disable minor (logarithmic) ticks
                colorbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=x_predictions.shape[0], decimals=1))

            heatmap_ax.set_xticks(bin_ticks, bins, rotation=90)
            heatmap_ax.set_yticks(bin_ticks, np.flip(bins), rotation=0)
            heatmap_ax.set_xlabel('Open / Close')
            heatmap_ax.set_ylabel('Wrist Rotation')

            if x_hist_ax is not None and y_hist_ax is not None:
                x_hist_axs.append(x_hist_ax)
                y_hist_axs.append(y_hist_ax)

                # Plot
                # nm_mask = np.all(np.abs(model_predictions) < bins[bins.shape[0] // 2], axis=1)
                # model_predictions = model_predictions[~nm_mask]
                x_hist_ax.hist(x_predictions, bins=bins)
                y_hist_ax.hist(y_predictions, bins=bins, orientation='horizontal')
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
            else:
                heatmap_ax.set_title(format_names(model))
                heatmap_ax.text(0, -0.25, f"{100 * simultaneity:.1f}%", # x and y values were set manually, so may require adjusting
                        ha='right', va='top', fontsize=12, fontweight='bold', bbox=bbox,
                        transform=heatmap_ax.transAxes)

        # Need to go through and align all histogram axes with eachother for consistent dimensions across subgrids (if applicable)
        for x_hist_ax, y_hist_ax in zip(x_hist_axs, y_hist_axs):
            x_hist_ax.sharey(x_hist_axs[0])
            y_hist_ax.sharex(y_hist_axs[0])

        fig.suptitle('Activation Heatmap')
        self._save_fig(fig, f"{stage}-heatmap.png")
        return fig

    def plot_loss(self):
        fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
        data = {
            'Epochs': [],
            'L1 Loss': [],
            'Model': []
        }
        for model in MODELS:
            for participant in self.participants:
                config = make_config(participant, model)
                loss_df = pd.read_csv(config.loss_file, header=None)
                batch_timestamps = loss_df.iloc[:, 0]
                batch_losses = loss_df.iloc[:, 1]
                epoch_timestamps = np.unique(batch_timestamps)
                assert (len(epoch_timestamps) == config.NUM_TRAIN_EPOCHS) or config.model_is_adaptive, f"Unexpected number of epochs in loss file. Expected {config.NUM_TRAIN_EPOCHS}, but got {len(epoch_timestamps)} for {participant.id} {model}."
                epoch_losses = [np.mean(batch_losses[batch_timestamps == timestamp]) for timestamp in epoch_timestamps]
                data['Epochs'].extend(list(range(len(epoch_timestamps))))
                data['L1 Loss'].extend(epoch_losses)
                data['Model'].extend(format_names([model for _ in range(len(epoch_timestamps))]))

        df = pd.DataFrame(data)
        sns.lineplot(df, x='Epochs', y='L1 Loss', hue='Model', ax=ax, palette=self.model_palette)
        ax.axvline(Config.NUM_TRAIN_EPOCHS, color='black', linestyle='--', label='SGT Epochs')
        text_y = ax.get_yticks()[-2]
        ax.annotate('SGT Training Epochs', 
             xy=(Config.NUM_TRAIN_EPOCHS, text_y),
             xytext=(Config.NUM_TRAIN_EPOCHS + 100, text_y),  # offset for better placement
             arrowprops=dict(arrowstyle='->', color='black'),
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
             ha='left',
             va='center',
             fontsize=10)
        ax.set_title('Training Loss')
        self._save_fig(fig, 'loss.png')
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
        self._save_fig(fig, 'offline.png')
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

    def plot_decision_stream(self, stage):
        fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
        log = self.read_log(self.participants[12], 'ciil', stage)

        decision_stream_predictions = []
        decision_stream_timestamps = []
        for trial in log.trials:
            decision_stream_predictions.extend(trial.predictions)
            decision_stream_timestamps.extend(trial.prediction_timestamps)
        decision_stream_predictions = np.array(decision_stream_predictions)

        if stage == ADAPTATION:
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
        elif stage == VALIDATION:
            ax.plot(decision_stream_timestamps, decision_stream_predictions[:, 1])
        else:
            self._raise_stage_error(stage)

        ax.set_xlabel('Timestamps')
        ax.set_ylabel('Pronation / Supination Activation')
        ax.set_title(f"Decision Stream ({stage})".title())
        self._save_fig(fig, f"{stage}-decision-stream.png")

    def plot_survey_results(self):
        if len(self.participants) == 1:
            print('Only 1 participant found - skipping plot of survey results.')
            return
        df = pd.read_csv(self.results_path.joinpath('intra-survey-results.csv'))
        questions = [ 
            'I had good control of isolated motions (e.g., move right)',
            'I had good control of combined motions (e.g., move up and right)',
            'I could stop and stay at rest when I wanted to',
            'I felt in control of the cursor',
            'I could control the speed well',
            'The training phase was engaging'
        ]
        palette = sns.color_palette('vlag')
        responses = {
            'Strongly agree': palette[0],
            'Agree': palette[1],
            'Neutral': palette[2],
            'Disagree': palette[-2],
            'Strongly disagree': palette[-1],
        }

        columns = list(responses.keys()) + ['Model', 'Question']
        
        if self.layout == PRESENTATION:
            fig, axs = plt.subplots(nrows=2, ncols=3, layout='constrained', figsize=(12, 4), sharex=True, sharey=True)
        else:
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
        self._save_fig(fig, 'survey.png')
        return fig

    def _save_fig(self, fig, filename):
        filepath = self.results_path.joinpath(filename)
        fig.savefig(filepath, dpi=self.dpi)
        print(f"File saved to {filepath.as_posix()}.")

    def _raise_stage_error(self, stage):
        raise ValueError(f"Unexpected value for stage. Current stage value: {stage}.")


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

    def extract_fitts_metrics(self, exclude_learning_trials = False):
        trials = self.trials
        if exclude_learning_trials:
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
            'simultaneity_gain': []
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
            fitts_results['simultaneity_gain'].append(t.calculate_simultaneity_gain())

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

    def calculate_simultaneity_gain(self):
        simultaneous_mask = np.all(np.abs(self.predictions) > 1e-3, axis=1)    # predictions are never exactly 0
        simultaneity = np.sum(simultaneous_mask) / self.predictions.shape[0]
        if simultaneity == 0:
            # Can't determine simultaneity gain for this trial - exclude it
            return np.nan
        return self.calculate_efficiency() / simultaneity


class ScreenshotFitts(libemg.environments.fitts.ISOFitts):
    def _draw(self):
        super()._draw()
        pygame.image.save(self.screen, Path(RESULTS_DIRECTORY).joinpath('fitts.png'))

    def _draw_cursor(self):
        color = self.config.cursor_in_target_color if self.dwell_timer is not None else self.config.cursor_color
        self._draw_circle(self.cursor, color, draw_radius=False)


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


def save_fitts_screenshot():
    # Start up the Fitts environment, take a screenshot, and close
    config = libemg.environments.fitts.FittsConfig(
        num_trials=1,
        width=1500,
        height=750,
        target_radius=TARGET_RADIUS,
        game_time=3,
        mapping='polar+',
        target_color=(0, 0, 0),
        background_color=(255, 255, 255)
    )
    controller = libemg.environments.controllers.KeyboardController()
    prediction_map = {
        pygame.K_UP: 'N',
        pygame.K_DOWN: 'S',
        pygame.K_RIGHT: 'E',
        pygame.K_LEFT: 'W',
        -1: 'NM'
    }
    fitts = ScreenshotFitts(controller, config, prediction_map=prediction_map, target_distance_radius=ISOFITTS_RADIUS)
    fitts.polygon_angles = np.linspace(0, 2 * math.pi, num=2000)
    fitts.run()
    # NOTE: The pygame window will likely stay open even after quitting and continuing on the main thread
    print('Fitts screenshot saved.')




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
    parser.add_argument('-l', '--layout', choices=(REPORT, PRESENTATION, THESIS), default=REPORT, help='Layout of plots. If not set, defaults to report format.')
    parser.add_argument('--fitts', action='store_true', help='Flag to launch Fitts window and store a screenshot.')
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

    if args.fitts:
        save_fitts_screenshot()

    calculate_participant_metrics(participants)
    plotter = Plotter(participants, layout=args.layout)
    plotter.plot_fitts_metrics(VALIDATION)
    plotter.plot_throughput_over_time()
    plotter.plot_dof_activation_heatmap(VALIDATION)
    plotter.plot_loss()
    plotter.plot_survey_results()
    
    plt.show()
    print('-------------Analysis complete!-------------')


if __name__ == '__main__':
    main()

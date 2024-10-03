import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

from emg_regression.metrics import check_input


class MidpointNormalize(Normalize):
    # Source: http://chris35wills.github.io/matplotlib_diverging_colorbar/
    def __init__(self, vmin = None, vmax = None, clip = False, midpoint = None):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip = None):
        x = [self.vmin, self.midpoint, self.vmax]
        y = [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def check_axis(ax):
    return plt.gca() if ax is None else ax


def format_title(title, replace_character = ' '):
    title = title.replace('-', replace_character)
    title = title.title()
    # Replace shorthand
    title = title.replace('Pro', 'Pronation')
    title = title.replace('Sup', 'Supination')
    return title


def plot_polyfit(x, y, ax):
    x = x.squeeze()
    y = y.squeeze()
    coefficients = np.polyfit(x, y, deg=2)
    p = np.poly1d(coefficients)
    ax.plot(x, p(x), '--k', label='Polynomial Fit')
    # Calculate R-squared
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum((y - p(x))**2)
    r_squared = 1 - (ss_residual / ss_total)
    ax.annotate(f'R2: {r_squared:.2f}', xy=(1, 0.2), xycoords='axes fraction', 
                 xytext=(-5, -5), textcoords='offset points', ha='right', va='top',
                 bbox=dict(boxstyle='round', alpha=0.1))


def plot_mav_vs_labels(mav, labels, ax, error = None, add_legend = False, add_polyfit = False, show_sections = False, title = '', color = 'black', label = '',
                       marker_size = 8):
    alpha = 0.4

    if show_sections:
        steady_state_threshold = 1e-5
        diff_threshold = 1e-3
        absolute_labels = np.abs(labels)
        no_motion_indices = absolute_labels <= steady_state_threshold
        steady_state_indices = np.abs(absolute_labels - 1) <= steady_state_threshold
        ramp_up_indices = np.diff(absolute_labels) > diff_threshold
        ramp_down_indices = np.diff(absolute_labels) < -diff_threshold

        # Append 0 to ramp up and ramp down for equal length
        ramp_up_indices = np.append(ramp_up_indices, False)
        ramp_down_indices = np.append(ramp_down_indices, False)
        steady_state_color = 'blue'
        ramp_color = 'red'
        sections_plot_config = [
            ('No Motion', 'X', no_motion_indices, steady_state_color),
            ('Steady State', 'o', steady_state_indices, steady_state_color),
            ('Ramp Up', '^', ramp_up_indices, ramp_color),
            ('Ramp Down', 'v', ramp_down_indices, ramp_color),
        ]
        for section_label, marker, indices, edgecolor in sections_plot_config:
            scatter = ax.scatter(labels[indices], mav[indices], c=color, s=marker_size, alpha=alpha, marker=marker, label=section_label, edgecolors=edgecolor)

        if error is not None:
            # Plot bands around MAV
            lower_error = np.copy(error)
            lower_bands = mav - error
            lower_error[lower_bands < 0] -= np.abs(lower_bands[lower_bands < 0])    # clip to 0 b/c MAV can't be less than 0
            error_bands = np.concatenate((lower_error.reshape(1, -1), error.reshape(1, -1)), axis=0)
            ax.errorbar(labels, mav, yerr=error_bands, color=color, label=label, alpha=0.2)

    else:
        scatter = ax.scatter(labels, mav, s=marker_size, alpha=alpha, c=np.linspace(0, 1, num=labels.shape[0]),
                                     label=label)
    if add_polyfit:
        plot_polyfit(labels, mav, ax)
    ax.set_title(title)
    if add_legend:
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))

    return scatter


def plot_mav_over_time(mav, labels, ax, sign_mav = True, title = '', mav_color='black'):
    def sign_vector(x):
        out = np.sign(x)
        zero_indices = np.abs(x) < 1E-2
        out[zero_indices] = 1
        return out

    max_mav = mav.max()
    if sign_mav:
        mav = mav * sign_vector(labels).reshape(-1, 1)

    mav_ax = ax
    labels_ax = ax.twinx()
    labels_color = 'blue'
    mav_ax.plot(mav, color=mav_color, alpha=0.1, lw=0.1)
    labels_ax.plot(labels, color=labels_color, label='Labels', alpha=0.25)

    # Plot formatting
    mav_ax.set_title(title)
    mav_ax.set_xlabel('Windows')
    mav_ax.set_ylabel('MAV', color=mav_color)
    mav_ax.set_ylim(bottom=-max_mav, top=max_mav)    # set y limits to max so it will be centered at 0 for clarity with other axis
    labels_ax.set_ylabel('Labels', color=labels_color)


def pca_transform(data):
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    pca = PCA(n_components=2)
    return pca.fit_transform(normalized_data)


def pca_plot(data, colour, title = '', r2 = None, ax = None, cmap = None, alpha = 1.0):
    if ax is None:
        ax = plt.gca()
    # Generate PCA plot
    pca_projection = pca_transform(data)
    scatter = ax.scatter(pca_projection[:, 0], pca_projection[:, 1], c=colour, cmap=cmap, alpha=alpha)
    # Formatting
    ax.set_title(title)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    if r2 is not None:
        ax.annotate(f'R2: {r2:.2f}', xy=(1, 1), xycoords='axes fraction', 
                     xytext=(-5, -5), textcoords='offset points', ha='right', va='top',
                     bbox=dict(boxstyle='round', alpha=0.1))
    return scatter


def plot_fft(signal, fs, ax, ignore_negative_frequencies = False, ignore_offset = False, title = ''):
    fourier = np.fft.fft(signal, axis=0)
    timestep = 1 / fs
    frequencies = np.fft.fftfreq(signal.shape[0], d=timestep)
    if ignore_negative_frequencies:
        half_signal_index = signal.shape[0] // 2
        fourier = fourier[:half_signal_index]
        frequencies = frequencies[:half_signal_index]
    if ignore_offset:
        fourier = fourier[1:]
        frequencies = frequencies[1:]
    ax.plot(frequencies, np.abs(fourier), alpha=0.25, lw=0.2)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('FFT')


def grouped_bar_plot(tick_groups, colour_groups, ax = None, bar_labels = None, horizontal = False, error = None, add_tick_labels = True):
    if ax is None:
        ax = plt.gca()
    x = np.arange(len(tick_groups))
    width = 0.25
    multiplier = 0

    for attribute, measurement in colour_groups.items():
        offset = width * multiplier
        err = error[multiplier] if error is not None else None
        if horizontal:
            rects = ax.barh(x + offset, measurement, width, label=attribute, xerr=err)
        else:
            rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=err)
        if bar_labels is not None:
            ax.bar_label(rects, labels=bar_labels[multiplier], padding=3, fontsize=10)
        multiplier += 1
    ticks = x + width / len(colour_groups)
    if not add_tick_labels:
        tick_groups = []
    if horizontal:
        ax.set_yticks(ticks, tick_groups)
    else:
        ax.set_xticks(ticks, tick_groups)


def plot_fitts_traces(traces, ax = None, title = '', color = 'blue', label = None):
    # Plot traces
    ax = check_axis(ax)
    for trace_idx, trace in enumerate(traces):
        ax.plot(trace[:, 0], trace[:, 1], color=color, linewidth=0.2, alpha=0.5)
        label = label if trace_idx == 0 else None
        ax.scatter(trace[0, 0], trace[0, 1], marker=f"${trace_idx}$", color=color, label=label)   # annotate start of trial
        ax.set_title(title)


def plot_embeddings(embeddings, labels, reducer, movement_mask = None, metrics = None, filename = '', cmap = 'cool', marker = '.', marker_size = 10, alpha = 1.0, label = ''):
    embeddings = check_input(embeddings)
    labels = check_input(labels)
    reduced_data = reducer.transform(embeddings)

    if movement_mask is not None:
        nrows = len(np.unique(movement_mask))
    else:
        nrows = labels.shape[1]
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, layout='constrained', figsize=(10, 10))


    for idx, ax in enumerate(axs):
        if movement_mask is not None:
            movement = np.unique(movement_mask)[idx]
            plot_data = reduced_data[movement_mask == movement]
            plot_labels = np.mean(labels[movement_mask == movement], axis=1)    # need a 1d vector, so take the mean
            title = f"Movement {movement}"
        else:
            plot_data = reduced_data
            plot_labels = labels[:, idx]
            title = f"DOF {idx}"
        scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1], marker=marker, s=marker_size, c=plot_labels, cmap=cmap, alpha=alpha, label=label)
        ax.set_title(title)
        ax.set_ylabel('Dim 2')
        
    
    fig.colorbar(scatter, ax=axs)
    axs[-1].set_xlabel('Dim 1')
        

    if metrics is not None:
        metrics_text = ''
        for metric, value in metrics.items():
            metrics_text += f"{metric}: {value:.2f}\n"
        axs[1].annotate(metrics_text, xy=(1, 1), xycoords='axes fraction', 
                     xytext=(-5, -5), textcoords='offset points', ha='right', va='top',
                     bbox=dict(boxstyle='round', alpha=0.1))
    if filename != '':
        fig.savefig(filename, dpi=400)
    return fig


def plot_residuals(predictions, labels):
    predictions = check_input(predictions)
    labels = check_input(labels)

    fig, axs = plt.subplots(nrows=2, ncols=labels.shape[1], sharex=True, sharey=True, layout='constrained', figsize=(10, 10))
    c = 'black'
    
    for dof_idx, (dof_predictions, dof_labels, dof_axs) in enumerate(zip(predictions.T, labels.T, axs.T)):
        residuals = dof_labels - dof_predictions
        dof_axs[0].scatter(dof_labels, dof_predictions, c=c, label='Predictions')
        dof_axs[0].plot(dof_labels, dof_labels, 'r--', label='Desired', linewidth=1)
        _, _, _, heatmap = dof_axs[1].hist2d(dof_labels, residuals, bins=20, range=[[labels.min(), labels.max()], [labels.min(), labels.max()]], cmap='coolwarm')

        dof_axs[0].set_title(f"DOF {dof_idx}")
        dof_axs[1].set_xlabel('Labels')


    axs[0, 0].set_ylabel('Predictions')
    axs[1, 0].set_ylabel('Residuals')
    axs[0, 0].legend(loc='upper left')
    fig.colorbar(heatmap, ax=axs[1], location='bottom', label='# of Occurrences')
    fig.suptitle('Residuals Plot')

    return fig


def plot_error_heatmap(predictions, labels):
    predictions = check_input(predictions)
    labels = check_input(labels)

    fig, ax = plt.subplots(layout='constrained', figsize=(10, 10))

    _, xbins, ybins = np.histogram2d(labels[:, 0], labels[:, 1], bins=20)

    xbin_mask = np.digitize(labels[:, 0], xbins)
    ybin_mask = np.digitize(labels[:, 1], ybins)
    errors_heatmap = []
    for xbin in np.unique(xbin_mask):
        xlabel_mask = xbin_mask == xbin
        row_errors = []
        for ybin in np.unique(ybin_mask):
            ylabel_mask = ybin_mask == ybin
            bin_labels = labels[xlabel_mask & ylabel_mask]
            bin_predictions = predictions[xlabel_mask & ylabel_mask]
            if len(bin_labels) == 0:
                error = 0
            else:
                error = np.mean(bin_labels - bin_predictions, axis=0).mean()
            row_errors.append(error)
        
        errors_heatmap.append(row_errors)

    errors_heatmap = np.array(errors_heatmap)
    max_error = np.abs(errors_heatmap).max()
    norm = MidpointNormalize(vmin=-max_error, vmax=max_error, midpoint=0.)
    im = ax.imshow(errors_heatmap, cmap='seismic', clim=(norm.vmin, norm.vmax), norm=norm)
    colorbar = fig.colorbar(im)
    colorbar.ax.set_ylabel('Mean residuals', rotation=-90, va='bottom')

    # Formatting
    ax.set_xlabel('DOF 0')
    ax.set_ylabel('DOF 1')
    ax.set_title('Error Heatmap')
    ax.set_xticks(np.arange(len(xbins)), labels=[f"{bin:.2f}" for bin in xbins])
    ax.set_yticks(np.arange(len(ybins)), labels=[f"{bin:.2f}" for bin in ybins])
    return fig


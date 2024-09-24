import os
import os.path as osp
import numpy as np
import pandas as pd

import torch
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.animation import FuncAnimation
from sklearn.metrics import RocCurveDisplay
from configs.config_global import FIG_DIR, ROOT_DIR

os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Add arial font, not necessary if you already have it
try:
    font_dirs = ["../arial-font"]  # The path to the custom font file.
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
except:
    pass

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Arial'

line_styles = ['-', '--', ':']
# Default colors
# colors = ['red', 'tomato', 'green', 'lightgreen', 'blue', 'lightblue']

if not osp.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


def adjust_figure(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout(pad=0.5)

def plot_chance_level(x_values, c_level=50):
    # chance level indicator
    plt.plot(x_values, len(x_values) * [c_level],
             color="r", linestyle='--', label='chance level')

def add_mod_suffix(model_name, mode):
    if model_name == 'pret1sConvRNNBL':
        suffix = ''
    else:
        suffix = mode
    return model_name + suffix

def get_sem(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

def make_rgb_with_alpha(rgb, alpha):
    """
    return a rgb color with alpha
    :param rgb: (r, g, b) in range [0, 1]
    :param alpha: alpha in range [0, 1]
    :return: (r, g, b) in range [0, 1]
    """
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, [1.0, 1.0, 1.0])]

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def line_plot(
    x_axis, 
    data_list,
    x_label,
    y_label,
    fig_dir,
    fig_name=None,
    label_list=[None, ] * 10,
    legend_title=None,
    xticks=None,
    yticks=None,
    linewidth=2,
    figsize=(6, 5),
    special_index=-1,
    legend=True,
    extra_lines=None
):
    
    plt.figure(figsize=figsize)

    for i, data, label in zip(range(len(data_list)), data_list, label_list):

        if i != special_index:

            plt.plot(
                x_axis, 
                data,
                label=label,
                linewidth=linewidth
            )
        else:
            
            plt.plot(
                x_axis,
                data,
                label=label,
                linestyle='dotted',
                linewidth=linewidth,
                marker='s',
                color='gray'
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if extra_lines is not None:
        extra_lines(
            plt, x_axis, 
            linewidth=linewidth
        )

    if legend:
        plt.legend(title=legend_title)

    if yticks is not None:
        plt.yticks(yticks)

    if xticks is not None:
        plt.xticks(xticks)
    
    adjust_figure()

    os.makedirs(osp.join(FIG_DIR, fig_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, fig_dir, f'{fig_name}.pdf'), transparent=True)
    plt.close()

def scatter_plot(
    X, Y,
    x_label,
    y_label,
    colors=None,
    dot_sizes=None,
    fig_dir='',
    legend_title=None,
    fig_name=None,
    xticks=None,
    yticks=None,
    fontsize=13,
    figsize=(3, 3),
    legend=False,
    legend_loc='best',
    legend_fontsize=None,
    extra_lines=None,
    alpha=None,
    xlim=None,
    ylim=None,
    marker='x'
):
    """
    A generic scatter plot
    :param X: An array representing the X axis
    :param Y: A list representing the Y axis
    :param colors: a list of colors for each data point
    :param extra_lines: a function that allows additional
        curves to be plotted, it should be defined by something like:
        def extra_lines(plt):
            plt(...)
    """

    plt.figure(figsize=figsize)
    plt.scatter(X, Y, s=dot_sizes, c=colors, alpha=alpha, marker=marker)

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)

    if extra_lines is not None:
        extra_lines(plt)

    if legend:
        if legend_fontsize is None:
            legend_fontsize = fontsize
        plt.legend(title=legend_title, loc=legend_loc,
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    if yticks is not None:
        plt.yticks(yticks)
    plt.yticks(fontsize=fontsize)

    if xticks is not None:
        plt.xticks(xticks)
    plt.xticks(fontsize=fontsize)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)
    
    adjust_figure()

    os.makedirs(osp.join(FIG_DIR, fig_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, fig_dir, f'{fig_name}.pdf'), transparent=True)
    plt.close()

def error_plot(
    x_axis, 
    data_list,
    x_label,
    y_label,
    label_list=[None, ],
    fig_dir='',
    legend_title=None,
    fig_name=None,
    xticks=None,
    yticks=None,
    linewidth=2,
    capsize=5,
    capthick=1.5,
    fontsize=13,
    figsize=(3.5, 3),
    legend=True,
    legend_frameon=False,
    legend_loc='best',
    legend_fontsize=None,
    legend_bbox_to_anchor=None,
    extra_lines=None,
    colors=None,
    mode='errorbar1',
    errormode='sem',
    alphas=None,
    y_offsets=None,
    xlim=None,
    ylim=None,
):
    """
    A generic error bar plot
    :param x_axis: A list representing the X axis
    :param data_list: A list of curves to plot,
        of size (num_curves, num_points, num_seed)
        Each curve is a list of the same length as x_axis,
        where each element in the list is a list of integers
        containing results of different random seeds
    :param label_list: the corresponding labels for the curves
    :param extra_lines: a function that allows additional
        curves to be plotted, it should be defined by something like:
        def extra_lines(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(...)
    :param mode: 'errorbar1', 'errorbar2', 'errorbar3', or 'errorshade'
    :param errormode: 'std' or 'sem'
    :param alphas: a list of alpha values for each curve
    :param y_offsets: a list of value to offset each curve
    """
    
    plt.figure(figsize=figsize)

    for i, data, label in zip(range(len(data_list)), data_list, label_list):

        mean = np.array([np.mean(val) for val in data])
        if y_offsets is not None:
            mean += y_offsets[i] * np.ones_like(mean)

        if errormode == 'std':
            error = np.array([np.std(val, ddof=1) for val in data])
        elif errormode == 'sem':
            error = np.array([get_sem(val) for val in data])
        else:
            raise NotImplementedError('error mode not implemented')

        if colors is not None:
            color = colors[i]
        else:
            color = f'C{i}'

        if alphas is not None:
            alpha = alphas[i]
        else:
            alpha = 1.0

        if mode == 'errorbar1':
            # error bar plot with single color
            plt.errorbar(
                x_axis,
                mean, error,
                label=label,
                linewidth=linewidth,
                capsize=capsize,
                capthick=capthick,
                fmt="o-",
                color=color,
                elinewidth=linewidth*0.6,
                markersize=linewidth*3.5,
                alpha=alpha
            )
        elif mode == 'errorbar2':
            # error bar plot with black outline
            plt.errorbar(
                x_axis,
                mean, error,
                label=label,
                linewidth=linewidth,
                capsize=capsize,
                capthick=capthick,
                fmt="o-",
                mfc=color,
                ecolor='k',
                color='k',
                elinewidth=linewidth*0.6,
                markersize=linewidth*4.5,
                alpha=alpha
            )
        elif mode == 'errorbar3':
            # error bar plot without cap
            plt.errorbar(
                x_axis,
                mean, error,
                label=label,
                linewidth=linewidth,
                fmt="o-",
                color=color,
                markeredgewidth=0,
                elinewidth=linewidth*0.6,
                markersize=linewidth*3.5,
                alpha=alpha
            )
        elif mode == 'errorshade':
            # show errors as shaded regions
            plt.plot(x_axis, mean, label=label, linewidth=linewidth,
                     color=color, alpha=alpha)
            plt.fill_between(
                x=x_axis,
                y1=mean - error,
                y2=mean + error,
                alpha=0.3*alpha,
                color=color,
                edgecolor=None,
            )
        else:
            raise NotImplementedError('Unknown mode')

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)

    if extra_lines is not None:
        extra_lines(
            plt, x_axis, 
            linewidth=linewidth, 
            capsize=capsize, capthick=capthick
        )

    if legend:
        if legend_fontsize is None:
            legend_fontsize = fontsize
        plt.legend(title=legend_title, loc=legend_loc, frameon=legend_frameon,
                   fontsize=legend_fontsize, title_fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor)

    if yticks is not None:
        plt.yticks(yticks)
    plt.yticks(fontsize=fontsize)

    if xticks is not None:
        plt.xticks(xticks)
    plt.xticks(fontsize=fontsize)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)
    
    adjust_figure()

    os.makedirs(osp.join(FIG_DIR, fig_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, fig_dir, f'{fig_name}.pdf'), transparent=True)
    plt.close()

def bar_plot(data, x_labels, y_label, label_list, fig_dir, fig_name, group_size=0.7, figsize=(4, 4)):
    """
    A grounped bar plot
    :param data: a 3d list of shape (len(label_list), len(x_labels), num_seeds) 
    """

    n = len(label_list)
    x = np.arange(n)  # the label locations
    width = group_size / n  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(label_list)):
        mean = [np.mean(x) for x in data[i]]
        ax.bar(x + ((i * 2 + 1) - n) / 2 * width, mean, width, label=label_list[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_xticks(x, x_labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    os.makedirs(osp.join(FIG_DIR, fig_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, fig_dir, f'{fig_name}.pdf'), transparent=True)
    plt.close()

def luckvogel_errorbar_plot(
    performance, 
    x_axis, 
    model_list, 
    plot_name, 
    xticks=(0, 4, 8, 12),
    yticks=(0.6, 0.8, 1.0),
    chance_level_label='Chance Level',
    plot_dir='luck_vogel',
    label_list=None,
    **kwargs
):

    def chance_level_line(plt, x_axis, linewidth, **unused):
        plt.hlines(
            0.5, np.min(x_axis), np.max(x_axis), 
            label=chance_level_label, 
            linestyle='dotted', color='r', 
            linewidth=linewidth
        )

    if label_list is None:
        label_list = model_list

    error_plot(
        x_axis,
        [performance[model] for model in model_list],
        'Set size',
        'Proportion correct',
        label_list,
        fig_name=plot_name,
        xticks=xticks,
        yticks=yticks,
        extra_lines=chance_level_line,
        fig_dir=plot_dir,
        **kwargs
    )


def continuous_report_plot(
    performance, 
    x_axis, 
    model_list,
    plot_name,
    y_label='Recall SD (deg)',
    plot_dir='continuous_report',
    fit_curve=False,
    show_curve_param=True,
    label_list=None,
    **kwargs
):
    
    from analysis.circular_stat import fit_power_law

    def fit_curves(plt, x_axis, linewidth, **unused):

        for model in model_list:
            mean = [np.mean(vals) for vals in performance[model]]
            if fit_curve:
                X, Y, popt = fit_power_law(x_axis, mean)
                if show_curve_param:
                    c, p = popt
                    label = "$y = {%.1f} x^{-%.1f}$" % (c, p)
                else:
                    label = None
                plt.plot(X, Y, color='black', linewidth=linewidth, label=label)

    if label_list is None:
        label_list = model_list

    error_plot(
        x_axis,
        [performance[model] for model in model_list],
        'Set Size',
        y_label,
        label_list,
        fig_name=plot_name,
        extra_lines=fit_curves,
        fig_dir=plot_dir,
        ylim=(0, None),
        **kwargs
    )


def sequential_performance_plot(
    performance,
    set_sizes,
    plot_name,
    plot_dir,
    plot_reciprocal=False,
    linewidth=2,
    capsize=5,
    capthick=1.5,
    fontsize=13,
    figsize=(4, 4),
    legend_title=None,
    ylabel='Recall std',
):

    plt.figure(figsize=figsize)
    n = max(set_sizes)
    plt.xticks(np.arange(-n, 0), labels=list(range(-n, -1)) + ['Last', ], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    for i, size in enumerate(set_sizes):

        if plot_reciprocal:
            mean = [(1 / np.array(vals)).mean() for vals in performance[i]]
            sem = [get_sem(1 / np.array(vals)) for vals in performance[i]]
        else:
            mean = [np.mean(val) for val in performance[i]]
            sem = [get_sem(val) for val in performance[i]]

        plt.errorbar(
            np.arange(-size, 0), 
            mean, sem, 
            label=size, 
            linewidth=linewidth, 
            capsize=capsize, 
            capthick=capthick
        )

    plt.legend(title=legend_title, fontsize=fontsize, title_fontsize=fontsize)

    plt.xlabel('Order in Sequence', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    adjust_figure()
    os.makedirs(osp.join(FIG_DIR, plot_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, plot_dir, f'{plot_name}.pdf'), transparent=True)
    plt.close()


def plot_density_fit_residual(x_axis, density, fit, residual,
                              title, save_name,
                              plot_dir='continuous_report_fit'):
    interval = x_axis[1] - x_axis[0]
    plt.figure(figsize=(5, 5))
    plt.bar(x_axis, density, label='Data', alpha=0.5, width=interval * 0.9)
    plt.plot(x_axis, fit, color='k', label='Mixture fit')
    plt.plot(x_axis, residual, label='Residual')
    plt.legend(frameon=False, fontsize=10)
    plt.xlabel('Estimation error (deg)')
    plt.ylabel('Density')
    plt.title(title)
    plt.ylim([-0.005, 0.03])
    adjust_figure()
    os.makedirs(osp.join(FIG_DIR, plot_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, plot_dir,
                         f'vonmises_uniform_fit_{save_name}.pdf'),
                transparent=True)
    plt.close()


def error_distribution_plot(
    errors, 
    label_list, 
    left, right, interval, 
    plot_name,
    plot_dir='continuous_report',
    x_label='Difference from target (deg)',
    y_label='Response frequency',
    mode='errorshade',
    coef=1, 
    **kwargs
):
    """
    :param errors: 3D list of shape (len(set_sizes), num_seeds, num_trials),
        the last dimension is a list of scalar errors
    """

    x_axis = list(np.arange(left, right + 0.01, interval))
    curve_list = []

    for error in errors:
        data_list = []
        
        for x in x_axis:
            densities = [((error_val > x - interval / 2) &
                          (error_val <= x + interval / 2)).mean() / interval * coef
                         for error_val in error]
            data_list.append(densities)
        
        curve_list.append(data_list)

    error_plot(
        x_axis,
        curve_list,
        x_label,
        y_label,
        label_list,
        fig_dir=plot_dir,
        fig_name=plot_name,
        figsize=(3.5, 3.5),
        mode=mode,
        **kwargs
    )

def error_range_plot(
    data_list,
    label_list,
    x_axis,
    plot_name, 
    fig_path, 
    x_label,
    y_label,
    plot_title=None, 
    legend=True,
    color_list=None,
    fill_vertical=None,
    font=14, 
    linewidth=2.5, 
    x_ticks=None,
    y_ticks=None,
    ylim=None,
    hline=None,
    fig_size=(4, 4)
):
    """
    A general plot for error range
    :param data_list: list of numpy arrays of shpae (n, len)
    """

    fig_path = osp.join(FIG_DIR, fig_path)
    plt.figure(figsize=fig_size)

    n = len(data_list)
    for i, (data, label) in enumerate(zip(data_list, label_list)):

        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))

        if color_list is not None:
            color = color_list[i]
        else:
            color = None

        plt.plot(x_axis, mean, label=label, linewidth=linewidth, color=color)
        plt.fill_between(
            x=x_axis,
            y1=mean - sem,
            y2=mean + sem,
            alpha=0.25,
            color=color
        )

    if fill_vertical is not None:
        plt.axvspan(-fill_vertical[0] - 0.5, fill_vertical[1] - 0.5, color='black', alpha=0.1)

    plt.xlabel(x_label, fontsize=font)
    plt.ylabel(y_label, fontsize=font)

    if x_ticks is not None:
        plt.xticks(x_ticks, fontsize=font)

    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=font)

    if hline is not None:
        plt.hlines(hline, min(x_ticks), max(x_ticks), color='black')

    if ylim is not None:
        plt.ylim(*ylim)

    if legend:
        plt.legend(loc='upper left', fontsize='small')

    if plot_title is not None:
        plt.title(plot_title, fontsize=font)

    adjust_figure()
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(osp.join(fig_path, f'{plot_name}.pdf'), transparent=True)
    plt.close()


def plot_gif(data, fig_path='', fig_name='', fig_title='', interval=200, start_idx = 0, **kwargs):
    """
    Draw a GIF from a list of tensors of shape (3, h, w) or (h, w)
    :param kwargs: additional keyword arguments will be passed to ax.imshow()
    """

    fig, ax = plt.subplots()

    def update(i):
        if data[i].dim() == 3:
            ax.imshow(data[i].cpu().permute(1, 2, 0).numpy(), **kwargs)
        else:
            ax.imshow(data[i].cpu().numpy(), **kwargs)
        ax.set_title(f"step = {i + start_idx}")
    
    ax.set_title(fig_title)

    if isinstance(data, torch.Tensor):
        num_frames = data.shape[0]
    else:
        num_frames = len(data)
    
    anim = FuncAnimation(fig, update, np.arange(num_frames), interval=interval)

    fig_path = osp.join(FIG_DIR, fig_path)
    os.makedirs(fig_path, exist_ok=True)
    anim.save(osp.join(fig_path, f'{fig_name}.gif'), dpi=80, writer='imagemagick')

    plt.close()

def plot_img(img, fig_path='', fig_name='', fig_title='', **kwargs):
    """
    Draw an image from torch.Tensor
    """

    fig, ax = plt.subplots()

    ax.set_title(fig_title)
    ax.imshow(img.detach().cpu().numpy(), **kwargs)

    fig_path = osp.join(FIG_DIR, fig_path)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(osp.join(fig_path, f'{fig_name}.pdf'), transparent=True)

    plt.close()


def plot_ROC_curves(scores_list, targets_list, set_sizes, fig_dir):
    """
    Plot ROC curves for different set sizes
    :param scores_list: list of list of scores, shape (num_set_sizes, num_trials)
    :param targets_list: list of list of targets, shape (num_set_sizes, num_trials)
    :param set_sizes: list of set sizes
    """
    cmap = cm.get_cmap('cool')
    colors = [cmap(1 - i) for i in np.linspace(0.0, 1.0, len(set_sizes))]
    
    plt.figure(figsize=(4, 4))
    fig, ax = plt.subplots(figsize=(4, 4))
    for i_ss, set_size in enumerate(set_sizes):
        RocCurveDisplay.from_predictions(
            targets_list[i_ss],
            scores_list[i_ss],
            name=f"{set_size}",
            color=colors[i_ss],
            ax=ax,
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.axis("square")
    plt.xlabel("False alarm rate", fontsize=14)
    plt.ylabel("Hit rate", fontsize=14)
    plt.xticks([0, 0.5, 1], fontsize=14)
    plt.yticks([0, 0.5, 1], fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.title("ROC curves of different set sizes")
    plt.legend(frameon=False, fontsize=12, title='Set Size', title_fontsize=12)
    
    adjust_figure()
    os.makedirs(osp.join(FIG_DIR, fig_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, fig_dir,
                         'ROC_all_seeds.pdf'), transparent=True)
    plt.close()

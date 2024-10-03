"""
Logger that wraps around wandb for convenience. Here is a description of each of the fields for wandb:
project: Project name. All experiments using the same dataset should be in the same project (e.g., paper).
name: Name of individual wandb.Run object. Should include some difference between the different run objects in a single script run (e.g., model). Will likely be changed in the UI depending on what makes the most sense for visuals.
may need to consider changing name to be more descriptive per model. things like comparing media shows across runs, so if multiple models have the same name then you can't show which is which. could just append the difference to the run input by CLI. Either that or leave as is and assume you can tidy that up in the dashboard if needed.
notes: Description of run. Should probably be CLI input from user describing the purpose of the run in more detail (e.g., comparing models X and Y b/c ...).
config: Configuration parameters. Dictionary of different parameters describing the individual wandb.Run object (e.g., num_epochs, model, ...).
tags: Tags that can be used to filter runs. This should be set in each script because each script should only generate runs for a single tag (i.e., tags can be used to identify scripts) (e.g., embedding-exploration).
job_type: Name to group together multiple run objects from a single script run. This should probably be a script parameter quickly marking what the purpose of this script run was (e.g., compare-model-x-y).

Tables are good for some things, but I think it's easier to just stick with logging without them. Make sure you log each type of image in a separate folder (e.g., images/model_type) so you can create a separate widget for each one. Logging metrics outside of tables allows you to control how they appear in the report better with run sets and you can view each step with a line view or the mean across with a bar plot view.
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from dataclasses import is_dataclass, asdict

import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import wandb

class Logger:
    def __init__(self, project, config, name = None, notes = None, tags = None, job_type = None, log = True) -> None:
        self.project = project
        self.config = config
        self.name = name
        self.notes = notes
        self.tags = tags
        self.job_type = job_type
        self.log = log
        self.run = None

    def __enter__(self):
        if not self.log:
            return
        assert wandb.run is None, 'Previous wandb run not terminated.'
        config = deepcopy(self.config)
        if is_dataclass(self.config) and not isinstance(self.config, type):
            # Convert to dictionary
            config = asdict(self.config)

        self.run = wandb.init(
            project=self.project,
            notes=self.notes,
            name=self.name,
            tags=self.tags,
            config=config,
            job_type=self.job_type
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finalize run
        wandb.finish()

    def log_value(self, value):
        if self.log:
            wandb.log(value)

    def log_figure(self, fig, label = '', caption = ''):
        if self.log:
            job_type = self.job_type if self.job_type is not None else ''
            key = Path('figures', job_type, label).as_posix()
            wandb.log({key: convert_plot_to_image(fig, caption=caption)})

    def log_model(self, model, path = None):
        if self.log:
            assert self.run is not None, 'Must start logger before logging.'
            path = Path(self.run.dir).joinpath('model.pt').as_posix() if path is None else path
            torch.save(model.state_dict(), path)
            self.run.log_model(path=path)

    def enable(self):
        self.log = True

    def disable(self):
        self.log = False

    def log_artifact(self, artifact):
        if self.log and self.run is not None:
            self.run.log_artifact(artifact)
        

def get_date_and_time():
    now = datetime.now()
    dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
    return dt_string


def convert_plot_to_image(fig, caption = ''):
    # Note: can also just pass in a matplotlib figure https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba_buffer = canvas.buffer_rgba()

    pil_image = Image.frombytes('RGBA', canvas.get_width_height(), rgba_buffer)
    return wandb.Image(pil_image, caption=caption)


def add_logging_cli_arguments(parser):
    # Eventually add parameter to let you test instead of train...
    parser.add_argument('--save_model', action='store_true', help='Flag to store model.')
    subparsers = parser.add_subparsers(title='Subcommands', description='Subcommands to determine script behaviour.', help='Run vs. debugging.',
                                       dest='mode', required=True)

    run_parser = subparsers.add_parser('run', help='Logged run.')
    run_parser.add_argument('job_type', type=str, help='Short name of script run used to group individual wandb.Run objects together.')

    subparsers.add_parser('debug', help='Debug mode, runs are not logged.')


def check_logging_arguments(args):
    args.log = args.mode != 'debug'
    if args.log:
        args.notes = input('Please enter a description of this run.\n') or None
    else:
        args.notes = None
        args.job_type = None

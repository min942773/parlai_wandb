#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Log metrics to tensorboard.

This file provides interface to log any metrics in tensorboard, could be
extended to any other tool like visdom.

.. code-block: none

   tensorboard --logdir <PARLAI_DATA/tensorboard> --port 8888.
"""

import os
import json
import numbers
from parlai.core.opt import Opt
from parlai.core.metrics import Metric
import parlai.utils.logging as logging

try:
    import wandb
except ImportError:
    wandb = None

class TensorboardLogger(object):
    """
    Log objects to tensorboard.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add tensorboard CLI args.
        """
        logger = argparser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-tblog',
            '--tensorboard-log',
            type='bool',
            default=False,
            help="Tensorboard logging of metrics, default is %(default)s",
            hidden=False,
        )

    def __init__(self, opt: Opt):
        try:
            # tensorboard is a very expensive thing to import. Wait until the
            # last second to import it.
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please run `pip install tensorboard tensorboardX`.')

        tbpath = opt['model_file'] + '.tensorboard'
        logging.debug(f'Saving tensorboard logs to: {tbpath}')
        if not os.path.exists(tbpath):
            os.makedirs(tbpath)
        self.writer = SummaryWriter(tbpath, comment=json.dumps(opt))

    def log_metrics(self, setting, step, report):
        """
        Add all metrics from tensorboard_metrics opt key.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        for k, v in report.items():
            if isinstance(v, numbers.Number):
                self.writer.add_scalar(f'{k}/{setting}', v, global_step=step)
            elif isinstance(v, Metric):
                self.writer.add_scalar(f'{k}/{setting}', v.value(), global_step=step)
            else:
                logging.error(f'k {k} v {v} is not a number')

    def flush(self):
        self.writer.flush()

class WandbLogger(object):
    """Log objects to Weights & Biases."""

    @staticmethod
    def add_cmdline_args(argparser):
        """Add wandb CLI args."""
        logger = argparser.add_argument_group('W&B Arguments')
        logger.add_argument(
            '-wblog',
            '--wandb-log',
            type='bool',
            default=False,
            help="W&B logging of metrics, default is %(default)s",
            hidden=False,
        )

    def __init__(self, opt):
        if wandb is None:
            raise ImportError('Please run `pip install wandb`.')

        wandb.init(anonymous="allow")

    def log_model(self, model):
        """
        Log model's gradients and parameters.
        :param model:
            The model to log
        """
        wandb.watch(model, log="all")

    def log_parameters(self, params):
        """
        Log the parameters used to train a model.
        :param params:
            The training parameters
        """
        wandb.config.update(params)

    def log_metrics(self, settings, step, report):
        """
        Log a report to W&B.
        :param settings:
            One of train/valid/test. Will be prepended to the metrics reported.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        annotated_report = {}
        for k, v in report.items():
            annotated_report[f'{settings}_{k}'] = v
        #print(annotated_report)
        #print(annotated_report['train_exs'])
        #print(float(annotated_report['train_exs']))

        wandb_report = {}
        for key in list(annotated_report.keys()):
            wandb_report[key] = float(annotated_report[key])

        wandb.log(wandb_report, step=step)

    def flush(self):
        wandb.Api().flush()
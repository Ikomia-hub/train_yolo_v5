# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.dnn import datasetio, dnntrain
import argparse
import copy
import os
import sys
# Your imports below
import yaml
import logging
import random
import time
import webbrowser
import subprocess
import atexit
import numpy as np
from warnings import warn
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from yolov5 import train as yolov5_train
#sys.path.insert(0, os.path.dirname(yolov5_train.__file__))

from yolov5.utils.general import check_file, check_git_status, fitness, get_latest_run, increment_path, print_mutation
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import plot_evolution
import YoloV5_dataset


logger = logging.getLogger()
wandb = None


def init_logging(rank=-1):
    if rank in [-1, 0]:
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        info = logging.StreamHandler(sys.stdout)
        info.setLevel(logging.INFO)
        info.setFormatter(formatter)
        logger.addHandler(info)

        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.ERROR)
        err.setFormatter(formatter)
        logger.addHandler(err)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARN)


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class YoloV5TrainParam(dataprocess.CDnnTrainProcessParam):

    def __init__(self):
        dataprocess.CDnnTrainProcessParam.__init__(self)
        self.dataset_folder = ""
        self.model_name = "yolov5s"
        self.epochs = 5
        self.batch_size = 16
        self.input_size = [512, 512]
        self.custom_hyp_file = ""
        self.launch_tensorboard = True
        self.output_folder = os.path.dirname(os.path.realpath(__file__)) + "/runs/"

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        super().setParamMap(paramMap)
        self.dataset_folder = paramMap["dataset_folder"]
        w = int(paramMap["input_width"])
        h = int(paramMap["input_height"])
        self.input_size = [w, h]
        self.custom_hyp_file = paramMap["custom_hyp_file"]
        self.launch_tensorboard = bool(paramMap["launch_tensorboard"])
        self.output_folder = paramMap["output_folder"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = super().getParamMap()
        param_map["dataset_folder"] = self.dataset_folder
        param_map["input_width"] = str(self.input_size[0])
        param_map["input_height"] = str(self.input_size[1])
        param_map["custom_hyp_file"] = self.custom_hyp_file
        param_map["launch_tensorboard"] = str(self.launch_tensorboard)
        param_map["output_folder"] = self.output_folder
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class YoloV5TrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        self.addInput(datasetio.IkDatasetIO())

        # Create parameters class
        if param is None:
            self.setParam(YoloV5TrainParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.opt = None
        self.tensorboard_proc = None
        # Terminate tensorboard process
        atexit.register(self.cleanup)

    def cleanup(self):
        self.tensorboard_proc.kill()

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.epochs
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        param = self.getParam()
        dataset_input = self.getInput(0)

        # Conversion from Ikomia dataset to YoloV5
        print("Preparing dataset...")
        dataset_yaml = YoloV5_dataset.prepare(dataset_input, param.dataset_folder, 0.9)

        print("Collecting configuration parameters...")
        self.opt = self.load_config(dataset_yaml)

        # Start TensorBoard server
        if param.launch_tensorboard:
            self.start_tensorboard(self.opt.project)

        print("Start training...")
        self.start_training()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def load_config(self, dataset_yaml):
        param = self.getParam()

        # Configuration options
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
        parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default='runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')

        config_path = os.path.dirname(os.path.realpath(__file__)) + "/config.yaml"

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            parser.set_defaults(**config)

        opt = parser.parse_args()
        opt.data = dataset_yaml

        # Override with GUI parameters
        if param.custom_hyp_file:
            opt.hyp = param.custom_hyp_file
        else:
            opt.hyp = os.path.dirname(yolov5_train.__file__) + "/" + opt.hyp

        opt.weights = param.model_name + ".pt"
        opt.epochs = param.epochs
        opt.batch_size = param.batch_size
        opt.img_size = param.input_size
        opt.project = param.output_folder
        opt.stop_train = False
        return opt

    def start_training(self):
        param = self.getParam()

        # Set DDP variables
        self.opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        self.opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        init_logging(self.opt.global_rank)

        if self.opt.global_rank in [-1, 0]:
            check_git_status()

        # Resume
        if self.opt.resume:  # resume an interrupted run
            ckpt = self.opt.resume if isinstance(self.opt.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            apriori = self.opt.global_rank, self.opt.local_rank

            with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
                self.opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace

            # reinstate
            self.opt.cfg, self.opt.weights, self.opt.resume, self.opt.global_rank, self.opt.local_rank = '', ckpt, True, *apriori
            logger.info('Resuming training from %s' % ckpt)
        else:
            # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
            # check files
            self.opt.data = check_file(self.opt.data)
            self.opt.cfg = check_file(self.opt.cfg)
            self.opt.hyp = check_file(self.opt.hyp)
            assert len(self.opt.cfg) or len(self.opt.weights), 'either --cfg or --weights must be specified'
            # extend to 2 sizes (train, test)
            self.opt.img_size.extend([self.opt.img_size[-1]] * (2 - len(self.opt.img_size)))
            self.opt.name = 'evolve' if self.opt.evolve else self.opt.name
            # increment run
            self.opt.save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok | self.opt.evolve)

        # DDP mode
        self.opt.total_batch_size = self.opt.batch_size
        device = select_device(self.opt.device, batch_size=self.opt.batch_size)

        if self.opt.local_rank != -1:
            assert torch.cuda.device_count() > self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            device = torch.device('cuda', self.opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert self.opt.batch_size % self.opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            self.opt.batch_size = self.opt.total_batch_size // self.opt.world_size

        # Hyperparameters
        with open(self.opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
            if 'box' not in hyp:
                warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                     (self.opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
                hyp['box'] = hyp.pop('giou')

        # Train
        logger.info(self.opt)
        if not self.opt.evolve:
            tb_writer = None  # init loggers
            if self.opt.global_rank in [-1, 0]:
                # Tensorboard
                if param.launch_tensorboard:
                    self.open_tensorboard()

                tb_writer = SummaryWriter(self.opt.save_dir)

            yolov5_train.train(hyp, self.opt, device, tb_writer, wandb, self.on_epoch_end)

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                    'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                    'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                    'box': (1, 0.02, 0.2),  # box loss gain
                    'cls': (1, 0.2, 4.0),  # cls loss gain
                    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                    'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                    'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                    'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                    'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                    'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                    'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                    'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                    'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                    'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

            assert self.opt.local_rank == -1, 'DDP mode not implemented for --evolve'
            self.opt.notest, self.opt.nosave = True, True  # only test/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            yaml_file = Path(self.opt.save_dir) / 'hyp_evolved.yaml'  # save best result here

            if self.opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % self.opt.bucket)  # download evolve.txt if exists

            for _ in range(300):  # generations to evolve
                if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights

                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([x[0] for x in meta.values()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)

                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)

                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = yolov5_train.train(hyp.copy(), self.opt, device, wandb=wandb)

                # Write mutation results
                print_mutation(hyp.copy(), results, yaml_file, self.opt.bucket)

            # Plot results
            plot_evolution(yaml_file)
            print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
                  f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

    def start_tensorboard(self, project_folder):
        if self.tensorboard_proc is None:
            cmd = ["tensorboard", "--logdir", project_folder]
            self.tensorboard_proc = subprocess.Popen(cmd)
            print("Waiting for TensorBoard to start...")
            time.sleep(5)

    def open_tensorboard(self):
        if self.tensorboard_proc is not None:
            url = "http://localhost:6006/"
            webbrowser.open(url, new=0)

    def on_epoch_end(self, metrics, step):
        # Step progress bar:
        self.emitStepProgress()
        # Log metrics
        metrics = self.conform_metrics(metrics)
        self.log_metrics(metrics, step)

    def stop(self):
        super().stop()
        self.opt.stop_train = True

    @staticmethod
    def conform_metrics(metrics):
        new_metrics = {}
        for tag in metrics:
            if "train/" in tag:
                val = metrics[tag].item()
            else:
                val = metrics[tag]

            tag = tag.replace(":", "-")
            new_metrics[tag] = val

        return new_metrics


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class YoloV5TrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "YoloV5Train"
        self.info.shortDescription = "Train Ultralytics YoloV5 object detection models."
        self.info.description = "Train Ultralytics YoloV5 object detection models."
        self.info.authors = "Ultralytics"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Train"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.year = 2020
        self.info.license = "GPLv3.0"
        # Code source repository
        self.info.repository = "https://github.com/ultralytics/yolov5"
        # Keywords used for search
        self.info.keywords = "train,object,detection,pytorch"

    def create(self, param=None):
        # Create process object
        return YoloV5TrainProcess(self.info.name, param)

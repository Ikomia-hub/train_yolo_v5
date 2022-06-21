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

from ikomia import dataprocess, core
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
import argparse
import copy
import os
import sys
import yaml
import logging
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from yolov5 import train as yolov5_train
from yolov5.utils.general import check_file, check_yaml, fitness, get_latest_run, increment_path, \
    print_mutation, colorstr
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import plot_evolve
from yolov5.utils.callbacks import Callbacks
from train_yolo_v5 import yolo_v5_dataset


_logger = logging.getLogger()
_plugin_folder = os.path.dirname(os.path.realpath(__file__))
_local_rank = int(os.getenv('LOCAL_RANK', -1))
_rank = int(os.getenv('RANK', -1))
_world_size = int(os.getenv('WORLD_SIZE', 1))


def init_logging(rank=-1):
    if rank in [-1, 0]:
        _logger.handlers = []
        _logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        info = logging.StreamHandler(sys.stdout)
        info.setLevel(logging.INFO)
        info.setFormatter(formatter)
        _logger.addHandler(info)

        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.ERROR)
        err.setFormatter(formatter)
        _logger.addHandler(err)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARN)


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class TrainYoloV5Param(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)

        # Create models folder
        models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        os.makedirs(models_folder, exist_ok=True)

        self.cfg["dataset_folder"] = ""
        self.cfg["model_name"] = "yolov5s"
        self.cfg["model_path"] = models_folder + os.sep + self.cfg["model_name"] + ".pt"
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 16
        self.cfg["input_width"] = 512
        self.cfg["input_height"] = 512
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["custom_hyp_file"] = ""
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"

    def setParamMap(self, param_map):
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["model_path"] = param_map["model_path"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["input_width"] = int(param_map["input_width"])
        self.cfg["input_height"] = int(param_map["input_height"])
        self.cfg["dataset_split_ratio"] = float(param_map["dataset_split_ratio"])
        self.cfg["custom_hyp_file"] = param_map["custom_hyp_file"]
        self.cfg["output_folder"] = param_map["output_folder"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class TrainYoloV5(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        if param is None:
            self.setParam(TrainYoloV5Param())
        else:
            self.setParam(copy.deepcopy(param))

        self.opt = None
        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        param = self.getParam()
        dataset_input = self.getInput(0)

        # Conversion from Ikomia dataset to YoloV5
        print("Preparing dataset...")
        print(param.cfg)
        dataset_yaml = yolo_v5_dataset.prepare(dataset_input, param.cfg["dataset_folder"], param.cfg["dataset_split_ratio"])

        print("Collecting configuration parameters...")
        self.opt = self.load_config(dataset_yaml)

        # Call beginTaskRun for initialization
        self.beginTaskRun()

        print("Start training...")
        self.start_training()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def load_config(self, dataset_yaml):
        param = self.getParam()

        if len(sys.argv) == 0:
            sys.argv = ["ikomia"]

        # Configuration options
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default='runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
        parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

        config_path = os.path.dirname(os.path.realpath(__file__)) + "/config.yaml"

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            parser.set_defaults(**config)

        opt = parser.parse_args(args=[])
        opt.data = dataset_yaml

        # Override with GUI parameters
        if param.cfg["custom_hyp_file"]:
            opt.hyp = param.cfg["custom_hyp_file"]
        else:
            opt.hyp = os.path.dirname(yolov5_train.__file__) + "/" + opt.hyp

        opt.weights = param.cfg["model_path"]
        opt.epochs = param.cfg["epochs"]
        opt.batch_size = param.cfg["batch_size"]
        opt.img_size = [param.cfg["input_width"], param.cfg["input_height"]]
        opt.project = param.cfg["output_folder"]
        opt.tb_dir = str(increment_path(Path(core.config.main_cfg["tensorboard"]["log_uri"]) / opt.name,
                                        exist_ok=opt.exist_ok))
        opt.stop_train = False

        if sys.platform == 'win32':
            opt.workers = 0

        return opt

    def start_training(self):
        callbacks = Callbacks()

        # Register callback for mlflow
        callbacks.register_action("on_fit_epoch_end", callback=self.on_epoch_end)

        # Resume
        if self.opt.resume and not self.opt.evolve:  # resume an interrupted run
            ckpt = self.opt.resume if isinstance(self.opt.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
                opt = argparse.Namespace(**yaml.safe_load(f))  # replace
            opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
            _logger.info(f'Resuming training from {ckpt}')
        else:
            self.opt.data, self.opt.cfg, self.opt.hyp, self.opt.weights, self.opt.project = \
                check_file(self.opt.data), check_yaml(self.opt.cfg), check_yaml(self.opt.hyp), \
                str(self.opt.weights), str(self.opt.project)  # checks
            assert len(self.opt.cfg) or len(self.opt.weights), 'either --cfg or --weights must be specified'
            if self.opt.evolve:
                self.opt.project = str(_plugin_folder / 'runs/evolve')
                self.opt.exist_ok, self.opt.resume = self.opt.resume, False  # pass resume to exist_ok and disable resume
            self.opt.save_dir = str(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))

        # DDP mode
        device = select_device(self.opt.device, batch_size=self.opt.batch_size)
        if _local_rank != -1:
            assert torch.cuda.device_count() > _local_rank, 'insufficient CUDA devices for DDP command'
            assert self.opt.batch_size % _world_size == 0, '--batch-size must be multiple of CUDA device count'
            assert not self.opt.image_weights, '--image-weights argument is not compatible with DDP training'
            assert not self.opt.evolve, '--evolve argument is not compatible with DDP training'
            torch.cuda.set_device(_local_rank)
            device = torch.device('cuda', _local_rank)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Train
        if not self.opt.evolve:
            yolov5_train.train(self.opt.hyp, self.opt, device, callbacks)
            if _world_size > 1 and _rank == 0:
                _logger.info('Destroying process group... ')
                dist.destroy_process_group()

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
                    'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                    'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(self.opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                    hyp['anchors'] = 3
            self.opt.noval, self.opt.nosave, save_dir = True, True, Path(self.opt.save_dir)  # only val/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
            if opt.bucket:
                os.system(f'gsutil cp gs://{self.opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

            for _ in range(self.opt.evolve):  # generations to evolve
                if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
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
                results = yolov5_train.train(hyp.copy(), self.opt, device, callbacks)

                # Write mutation results
                print_mutation(results, hyp.copy(), save_dir, self.opt.bucket)

            # Plot results
            plot_evolve(evolve_csv)
            print(f'Hyperparameter evolution finished\n'
                  f"Results saved to {colorstr('bold', save_dir)}\n"
                  f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')

    def on_epoch_end(self, vals, epoch, best_fitness, fi):
        # Step progress bar:
        self.emitStepProgress()
        # Log metrics
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        metrics = self.conform_metrics(x)
        self.log_metrics(metrics, epoch)

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
class TrainYoloV5Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_yolo_v5"
        self.info.shortDescription = "Train Ultralytics YoloV5 object detection models."
        self.info.description = "Train Ultralytics YoloV5 object detection models."
        self.info.authors = "Ultralytics"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
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
        return TrainYoloV5(self.info.name, param)

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
# Add yolov5 git submodule to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/yolov5")
# Your imports below
import yaml
import logging
import random
import time
import numpy as np
from warnings import warn
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from yolov5 import train as yolov5_train
from yolov5.utils.general import check_file, check_git_status, fitness, get_latest_run, increment_path, \
    print_mutation
from yolov5.utils.torch_utils import select_device
from utils.plots import plot_evolution
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
        self.output_folder = os.path.dirname(os.path.realpath(__file__)) + "/runs/"

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.dataset_folder = paramMap["dataset_folder"]
        self.output_folder = paramMap["output_folder"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = core.ParamMap()
        paramMap["dataset_folder"] = self.dataset_folder
        paramMap["output_folder"] = self.output_folder
        return paramMap


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

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
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
        self.emitStepProgress()
        print("Collecting configuration parameters...")
        opt = self.load_config(dataset_yaml)
        self.emitStepProgress()
        print("Start training...")
        self.start_training(opt)

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def load_config(self, dataset_yaml):
        param = self.getParam()

        # Configuration options
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default='yolov5/data/coco128.yaml', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='yolov5/data/hyp.scratch.yaml', help='hyperparameters path')
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
        print(config_path)

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            parser.set_defaults(**config)

        opt = parser.parse_args()
        opt.data = dataset_yaml
        opt.hyp = os.path.dirname(os.path.realpath(__file__)) + "/" + opt.hyp
        opt.project = param.output_folder
        return opt

    def start_training(self, opt):
        # Set DDP variables
        opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        init_logging(opt.global_rank)

        if opt.global_rank in [-1, 0]:
            check_git_status()

        # Resume
        if opt.resume:  # resume an interrupted run
            ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            apriori = opt.global_rank, opt.local_rank

            with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
                opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace

            opt.cfg, opt.weights, opt.resume, opt.global_rank, opt.local_rank = '', ckpt, True, *apriori  # reinstate
            logger.info('Resuming training from %s' % ckpt)
        else:
            # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
            opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
            opt.name = 'evolve' if opt.evolve else opt.name
            opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

        # DDP mode
        opt.total_batch_size = opt.batch_size
        device = select_device(opt.device, batch_size=opt.batch_size)

        if opt.local_rank != -1:
            assert torch.cuda.device_count() > opt.local_rank
            torch.cuda.set_device(opt.local_rank)
            device = torch.device('cuda', opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            opt.batch_size = opt.total_batch_size // opt.world_size

        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
            if 'box' not in hyp:
                warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                     (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
                hyp['box'] = hyp.pop('giou')

        # Train
        logger.info(opt)
        if not opt.evolve:
            tb_writer = None  # init loggers
            if opt.global_rank in [-1, 0]:
                logger.info(
                    f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
                tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
            yolov5_train.train(hyp, opt, device, tb_writer, wandb)

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

            assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
            opt.notest, opt.nosave = True, True  # only test/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here

            if opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

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
                results = yolov5_train.train(hyp.copy(), opt, device, wandb=wandb)

                # Write mutation results
                print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

            # Plot results
            plot_evolution(yaml_file)
            print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
                  f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class YoloV5TrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "YoloV5Train"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return YoloV5TrainProcess(self.info.name, param)

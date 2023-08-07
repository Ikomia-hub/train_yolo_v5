import logging
from ikomia.utils.tests import run_for_test
import os

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train yolo v5 =====")
    input_dataset = t.get_input(0)
    params = t.get_parameters()
    dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(dataset_folder, exist_ok=True)
    params["epochs"] = "2"
    params["batch_size"] = "1"
    params["dataset_folder"] = dataset_folder
    params["dataset_split_ratio"] = "0.5"
    t.set_parameters(t)
    input_dataset.load(data_dict["datasets"]["detection"]["dataset_coco"])
    yield run_for_test(t)

# Copyright (c) Facebook, Inc. and its affiliates.
from .cityscapes_evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator
from .coco_evaluation import COCOEvaluator
from .rotated_coco_evaluation import RotatedCOCOEvaluator
from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_context, inference_on_dataset
from .lvis_evaluation import LVISEvaluator
from .panoptic_evaluation import COCOPanopticEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .sem_seg_evaluation import SemSegEvaluator
from .testing import print_csv_format, verify_results
from .text_evaluation import TextEvaluator
from .text_eval_script import text_eval_main
from . import rrc_evaluation_funcs

__all__ = [k for k in globals().keys() if not k.startswith("_")]

from detectron2.config import CfgNode as CN


def add_SWINTS_config(cfg):
    """
    Add config for SWINTS.
    """
    cfg.MODEL.SWINTS = CN()
    cfg.MODEL.SWINTS.NUM_CLASSES = 80
    cfg.MODEL.SWINTS.NUM_PROPOSALS = 300
    cfg.MODEL.SWINTS.TEST_NUM_PROPOSALS = 100

    # RCNN Head.
    cfg.MODEL.SWINTS.NHEADS = 8
    cfg.MODEL.SWINTS.DROPOUT = 0.0
    cfg.MODEL.SWINTS.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SWINTS.ACTIVATION = 'relu'
    cfg.MODEL.SWINTS.HIDDEN_DIM = 256
    cfg.MODEL.SWINTS.NUM_CLS = 3
    cfg.MODEL.SWINTS.NUM_REG = 3
    cfg.MODEL.SWINTS.NUM_MASK = 3
    cfg.MODEL.SWINTS.NUM_HEADS = 6

    cfg.MODEL.SWINTS.MASK_DIM = 60


    # Dynamic Conv.
    cfg.MODEL.SWINTS.NUM_DYNAMIC = 2
    cfg.MODEL.SWINTS.DIM_DYNAMIC = 64

    # Recognition Head
    cfg.MODEL.REC_HEAD = CN()
    cfg.MODEL.REC_HEAD.BATCH_SIZE = 48
    cfg.MODEL.REC_HEAD.POOLER_RESOLUTION = (28,28)
    cfg.MODEL.REC_HEAD.RESOLUTION = (32, 32)
    cfg.MODEL.REC_HEAD.NUM_CLASSES = 107

    # Loss.
    cfg.MODEL.SWINTS.CLASS_WEIGHT = 2.0
    cfg.MODEL.SWINTS.GIOU_WEIGHT = 2.0
    cfg.MODEL.SWINTS.L1_WEIGHT = 5.0
    cfg.MODEL.SWINTS.REC_WEIGHT = 1.0
    cfg.MODEL.SWINTS.DEEP_SUPERVISION = True
    cfg.MODEL.SWINTS.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SWINTS.MASK_WEIGHT = 2.0

    # Focal Loss.
    cfg.MODEL.SWINTS.ALPHA = 0.25
    cfg.MODEL.SWINTS.GAMMA = 2.0
    cfg.MODEL.SWINTS.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # Matcher
    cfg.MODEL.SWINTS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.SWINTS.IOU_LABELS = [0, 1]

    # Encoder
    cfg.MODEL.SWINTS.PATH_COMPONENTS = "./projects/SWINTS/LME/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60_siz28.npz"
    
    # SWINT backbone
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addation
    cfg.MODEL.FPN.TOP_LEVELS = 2

    # Test config
    cfg.TEST.USE_NMS_IN_TSET = True
    cfg.TEST.INFERENCE_TH_TEST = 0.4
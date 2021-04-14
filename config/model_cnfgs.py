class Cfgs():
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_AUX_FEAT = False
        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = True
        self.WORD_EMBED_SIZE = 300

        self.CKPTS_PATH = 'models' 

        self.DATASET = 'vqa'

        self.FEAT_SIZE = {
            'vqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            }
        }

        # Set if bbox_feat need be normalize by image size, default: False
        self.BBOX_NORMALIZE = False

        # Default training batch size: 64
        self.BATCH_SIZE = 64

        # Multi-thread I/O
        self.NUM_WORKERS = 8

        # Use pin memory
        # (Warning: pin memory can accelerate GPU loading but may
        # increase the CPU memory usage when NUM_WORKS is big)
        self.PIN_MEM = True

        # Large model can not training with batch size 64
        # Gradient accumulate can split batch to reduce gpu memory usage
        # (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
        self.GRAD_ACCU_STEPS = 1

        # The base learning rate
        self.LR_BASE = 0.0001

        # Learning rate decay ratio
        self.LR_DECAY_R = 0.2

        # Learning rate decay at {x, y, z...} epoch
        self.LR_DECAY_LIST = [10, 12]

        # Warmup epoch lr*{1/(n+1), 2/(n+1), ... , n/(n+1)}
        self.WARMUP_EPOCH = 3

        # Max training epoch
        self.MAX_EPOCH = 13

        # Gradient clip
        # (default: -1 means not using)
        self.GRAD_NORM_CLIP = -1

        # OPT
        self.OPT = 'Adam'
        self.OPT_PARAMS = {'betas': (0.9, 0.98), 'eps': 1e-09, 'weight_decay': 0, 'amsgrad': False}
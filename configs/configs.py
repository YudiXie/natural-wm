"""
Configurations for the project
format adapted from https://github.com/gyyang/olfaction_evolution

Be aware that the each field in the configuration must be in basic data type that
jason save and load can preserve. each field cannot be complicated data type
"""
import numpy as np

class BaseConfig(object):
    def __init__(self):
        """
        model_type: model type, eg. "ConvRNNBL"
        task_type: task type, eg. "n_back"
        """
        self.experiment_name = None
        self.model_name = None
        self.save_path = None
        self.task_type = None
        self.config_mode = 'train'
        self.seed = 0

        # basic training parameters
        self.batch_size = 128
        self.optimizer_type = 'AdamW'
        self.grad_clip = 1.0 # max norm of grad clipping, eg. 1.0 or None
        self.use_lr_scheduler = False
        self.scheduler_type = None
        self.lr = 0.001
        self.lr_SGD = 0.1
        self.wdecay = 0
        
        self.num_ep = 1000 # num of epochs, usually not used
        self.max_batch = 20000
        self.log_every = 1000
        self.save_every = 100000
        self.test_batch = 100

        # only used in classification pretraining
        self.eslen = 5
        self.early_stop = False
        self.perform_test = True # if test protocol is the same as train
        self.inc_targets = None
        self.model_class_size = 10 # output size of the model, which is the number of classes in classification tasks
        self.joint_train = False
        self.mod_w = [1.0]  # weights for each of the modality, must sum to 1
        self.modality = None

        self.overwrite = True
        self.neurogym = True
        self.input_resolution = None # input would be resized to this resolution, by deault no resizing
        self.input_noise_resolution = None
        self.input_noise = 0.1
        self.input_noise_mode = 'per_step' # per_period, per_step or per_trial, see datasets/dataloader.py
        self.curriculum = {}
        self.print_mode = 'accuracy' # or print error

        # Specify required resources (useful when running on cluster)
        self.hours = 12
        self.mem = 16
        self.cpu = 2
        self.num_workers = 1        
        self.gpu_constraint = 'high-capacity' # Refer to https://github.mit.edu/MGHPCC/OpenMind/wiki/How-to-submit-GPU-jobs%3F

        # if not None, load model from the designated path
        self.load_path = None
        self.strict_loading = True # whether to enforce that loaded models have the same keys

        # default model config
        self.model_type = 'AttCNNtoRNN'
        self.rnn = 'CTRNN'
        self.hidden_size = 256
        self.att = 'cbam'
        self.att_layers = (0, 1, 2)
        self.save_cnn = False
        self.freeze_rnn = False
        self.dt = 50
        self.rnn_eta = 200 # time constant for CTRNN
        self.layernorm = False 
        self.rnn_noise = 0
        self.additive_rnn_noise = 0.05
        self.use_pos_encoding = False # whether to add positional encoding to CNN embedding

        # config for CNN model
        self.cnn_width = 64 # dim of final feature map, should be a multiple of 4
        self.embedding_size = None # if not None, only use the first embedding_size dimensions
        self.cnn_norm = 'layernorm'  # batchnorm, instancenorm, none
        self.cnn_archi = 'ResNet'
        self.cnn_pret = 'Classification_CIFAR10'
        self.freeze_cnn = True
       
        # whether to use different pretrain models on different random seeds
        self.different_pretrain_seeds = True
        # ResNet config
        self.spatial_average = True
        self.resblock_config = [3, 3, 3]

        # for task functions, deprecated
        self.pretrain_dataset = 'CIFAR10'
        self.dataset = 'MNIST'
        
        
    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class DMSConfig(BaseConfig):
    def __init__(self):
        super(DMSConfig, self).__init__()

        self.early_stop = False
        self.max_batch = 10000
        self.batch_size = 64

        # task config
        self.task_type = 'delayed_match'
        # TODO: n_back attribute should be removed in future
        self.n_back = 1
        self.sample_step = 5
        self.delay_step = (0, 8)
        self.test_step = 5
        self.distortion = None
        self.add_noise = True

        # shared model and task config
        self.model_class_size = 1
        self.comp = True

class LuckVogelConfig(DMSConfig):

    def __init__(self):
        super().__init__()

        # Default task setting
        self.dataset = "LuckVogel"
        self.task_type = 'change_detection'
        self.img_size = (3, 32, 32)
        self.lg_patch_size = (5, 5)
        self.num_patches = (1, 2, 3, 4, 8, 12)
        self.use_fixed_colors = True
        self.change_magnitude = np.pi

        # self.lg_small_patch_size = (3, 3) # not used in ngym version
        # self.delay_step = 5
        
        # Default training setting
        self.max_batch = 30000
        self.hours = 12

        self.max_pretrain_set_size = 12
        self.curriculum = {
            '0': dict(easy_mode=True),
            '10000': dict(easy_mode=False)
        }

class ContinuousReportConfig(BaseConfig):

    def __init__(self):
        super().__init__()

        # task config
        self.dataset = 'ContinuousReportDataset'
        self.task_type = 'continuous_report'
        self.img_size = (3, 32, 32)
        self.patch_size = (5, 5)
        self.possible_colors = 360
        self.num_patches = (1, 2, 3, 4, 5, 6)
        self.activity_regularization = 0
        self.fixed_positions = None
        self.fixed_color_angles = None
        self.minimal_pairwise_distance = np.pi / 12
        self.output_uncertainty = False
        self.print_mode = 'error'

        # Default training setting
        self.max_batch = 60000
        self.hours = 24

class SequentialContinuousReportConfig(ContinuousReportConfig):

    def __init__(self):
        super().__init__()

        # task config
        self.dataset = 'SequentialContinuousReportDataset'
        self.task_type = 'sequential_continuous_report'
        # self.task_type = 'SequentialContinuousReport'

        # Timing configs currently only works for task function version
        # TODO: add configs for neurogym version
        self.sample_step = 2
        self.interval_step = 1
        self.delay_step = 2
        self.test_step = 5
        
        self.num_patches = (1, 2, 3, 4, 5, 6)
        self.gpu_constraint = '18GB'
        self.hours = 36
        self.max_batch = 60000

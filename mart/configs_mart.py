"""
Definition of constants and configurations for captioning with MART.
"""

from typing import Any, Dict, Optional

from nntrainer import trainer_configs
from nntrainer.typext import ConstantHolder


# ---------- Path constants ----------
class MartPathConst(ConstantHolder):
    CACHE_DIR = "cache_caption"
    COOT_FEAT_DIR = "provided_embeddings"
    ANNOTATIONS_DIR = "annotations"
    VIDEO_FEATURE_DIR = "data/mart_video_feature"


# ---------- MART config ----------
class MartDatasetConfig(trainer_configs.BaseDatasetConfig):
    """
    MART Dataset Configuration class

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.preload: bool = config.pop("preload")


class MartConfig(trainer_configs.BaseExperimentConfig):
    """
    Definition to load the yaml config files for training a MART captioning model.
    This is where the actual config dict goes and is processed.


    Args:
        config: Configuration dictionary to be loaded, logging part.

    Attributes:
        train: Training config
        val: Validation config
        dataset_train: Train dataset config
        dataset_val: Val dataset config
        logging: Log frequency

        max_n_sen: for recurrent, max number of sentences, 6 for anet, 12 for yc2
        max_n_sen_add_val: Increase max_n_sen during validation (default 10)
        max_t_len: max length of text (sentence or paragraph), 30 for anet, 20 for yc2
        max_v_len: max length of video feature (default 100)
        type_vocab_size: Size of sequence type vocabulary: video as 0, text as 1 (default 2)
        word_vec_size: GloVE embeddings size (default 300)

        coot_model_name: Model name to load embeddings from. (default None)
        coot_mode: Which COOT representations to input into the captioning model. (default vidclip, choices
            vid: only video, vidclip: video + clip, vidclipctx: video + clip + video context, clip: only clip)
        coot_dim_vid: video feature size (default 768)
        coot_dim_clip: clip feature size (default 384)
        video_feature_size: 2048 appearance + 1024 flow. Change depending on COOT embeddings:
            vidclip: coot_dim_vid + coot_dim_clip, clip: coot_dim_clip, etc. (default 3072)

        debug: Activate debugging. Unused / untested (default False)

        attention_probs_dropout_prob: Dropout on attention mask (default 0.1)
        hidden_dropout_prob: Dropout on hidden states (default 0.1)
        hidden_size: Model hidden size (default 768)
        intermediate_size: Model intermediate size (default 768)
        layer_norm_eps: Epsilon parameter for Layernorm (default 1e-12)
        max_position_embeddings: Position embeddings limit (default 25)
        memory_dropout_prob: Dropout on memory cells (default 0.1)
        num_attention_heads: Number of attention heads (default 12)
        num_hidden_layers: number of transformer layers (default 2)
        n_memory_cells: number of memory cells in each layer (default 1)
        share_wd_cls_weight: share weight matrix of the word embedding with the final classifier (default False)
        recurrent: Run recurrent model (default False)
        untied: Run untied model (default False)
        mtrans: Masked transformer model for single sentence generation (default False)
        xl: transformer xl model, enforces recurrent = True, since the data loading part is the same (default False)
        xl_grad: enable back-propagation for xl model, only useful when `-xl` flag is enabled.
            Note, the original transformerXL model does not allow back-propagation. (default False)
        use_glove: Disable loading GloVE embeddings. (default None)
        freeze_glove: do not train GloVe vectors (default False)
        model_type: This is inferred from the fields recurrent, untied, mtrans, xl, xl_grad

        label_smoothing: Use soft target instead of one-hot hard target (default 0.1)

        save_mode: all: save models at each epoch. best: only save the best model (default best, choices: all, best)
        use_beam: use beam search, otherwise greedy search (default False)
        beam_size: beam size (default 2)
        n_best: stop searching when get n_best from beam search (default 1)

        ema_decay: Use exponential moving average at training, float in (0, 1) and -1: do not use.
            ema_param = new_param * ema_decay + (1-ema_decay) * last_param (default 0.9999)
        initializer_range: Weight initializer range (default 0.02)
        lr: Learning rate (default 0.0001)
        lr_warmup_proportion: Proportion of training to perform linear learning rate warmup for.
            E.g., 0.1 = 10%% of training. (default 0.1)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.name = "config_ret"

        # mandatory groups, needed for nntrainer to work correctly
        self.train = trainer_configs.BaseTrainConfig(config.pop("train"))
        self.val = trainer_configs.BaseValConfig(config.pop("val"))
        self.dataset_train = MartDatasetConfig(config.pop("dataset_train"))
        self.dataset_val = MartDatasetConfig(config.pop("dataset_val"))
        self.logging = trainer_configs.BaseLoggingConfig(config.pop("logging"))
        self.saving = trainer_configs.BaseSavingConfig(config.pop("saving"))

        # more training
        self.label_smoothing: float = config.pop("label_smoothing")

        # more validation
        self.save_mode: str = config.pop("save_mode")
        self.use_beam: bool = config.pop("use_beam")

        # dataset
        self.max_n_sen: int = config.pop("max_n_sen")
        self.max_t_len: int = config.pop("max_t_len")
        self.max_v_len: int = config.pop("max_v_len")
        self.num_img: int = config.pop("num_img")

        # technical
        self.debug: bool = config.pop("debug")

        # model
        self.hidden_size: int = config.pop("hidden_size")
        self.enc_num_layers: int = config.pop("enc_num_layers")
        self.dec_num_layers: int = config.pop("dec_num_layers")
        self.img_feat_size: int = config.pop("img_feat_size")
        self.word_feat_size: int = config.pop("word_feat_size")
        self.vocab_size: int = config.pop("vocab_size")
        self.max_seq_length: int = config.pop("max_seq_length")
        self.use_glove: bool = config.pop("use_glove")
        self.freeze_glove: bool = config.pop("freeze_glove")

        # optimization
        self.ema_decay: float = config.pop("ema_decay")
        self.initializer_range: float = config.pop("initializer_range")
        self.lr: float = config.pop("lr")
        self.lr_warmup_proportion: float = config.pop("lr_warmup_proportion")

        # max position embeddings is calculated as the max joint sequence length
        self.max_position_embeddings: int = self.max_v_len + self.max_t_len

        # must be set manually as it depends on the dataset
        self.vocab_size: Optional[int] = None

        self.recurrent = True
        self.xl_grad = self.xl = False

        # infer model type
        if self.recurrent:  # recurrent paragraphs
            if self.xl:
                if self.xl_grad:
                    self.model_type = "xl_grad"
                else:
                    self.model_type = "xl"
            else:
                self.model_type = "re"
        else:  # single sentence
            if self.untied:
                self.model_type = "untied_single"
            elif self.mtrans:
                self.model_type = "mtrans_single"
            else:
                self.model_type = "single"

        self.post_init()


class MartMetersConst(ConstantHolder):
    """
    Additional metric fields.
    """

    TRAIN_LOSS_PER_WORD = "train/loss_word"
    TRAIN_ACC = "train/acc"

    VAL_LOSS_PER_WORD = "val/loss_word"
    VAL_ACC = "val/acc"
    GRAD = "train/grad"

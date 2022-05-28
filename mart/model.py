"""
MART model.

"""
import copy
import logging
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard.summary import video
import torchvision.models as models

from mart.configs_mart import MartConfig, MartPathConst
from mart.masked_transformer import MTransformer
from mart.loss_caption import LabelSmoothingLoss
from nntrainer.utils_torch import count_parameters
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# # default infinity (cfg.inf = 0), works with fp32. this can lead to NaN values in some circumstances
INF = float("inf")





def create_mart_model(
    cfg: MartConfig,
    vocab_size: int,
    cache_dir: str = MartPathConst.CACHE_DIR,
    verbose: bool = True,
) -> nn.Module:
    """
    Args:
        cfg: Experiment cfg.
        vocab_size: Vocabulary, calculated in mart as len(train_set.word2idx).
        cache_dir: Cache directory.
        verbose: Print model name and number of parameters.

    Returns:
        MART model.
    """
    cfg.vocab_size = vocab_size
    if cfg.recurrent:
        logger.info("Use recurrent model - Mine")
        model = RecursiveTransformer(cfg)
    if cfg.use_glove:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(
                    torch.load(
                        Path(cache_dir) / f"{cfg.dataset_train.name}_vocab_glove.pt"
                    )
                ).float(),
                freeze=cfg.freeze_glove,
            )
        else:
            logger.warning(
                "This model has no embeddings, cannot load glove vectors into the model"
            )

    # output model properties
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        count_parameters(model)
        if hasattr(model, "embeddings") and hasattr(
            model.embeddings, "word_embeddings"
        ):
            count_parameters(model.embeddings.word_embeddings)

    return model


class RegressionNet(torch.nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 8, 2)
        self.conv2 = torch.nn.Conv2d(16, 64, 8, 2)
        self.conv3 = torch.nn.Conv2d(64, 256, 8, 2)
        self.conv4 = torch.nn.Conv2d(256, 512, 8, 2)
        self.conv5 = torch.nn.Conv2d(512, 1024, 8, 1)


    def forward(self, x): 
        x = x.permute(0, 3, 1, 2) # input: (batch_size, 3, 224, 224)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x).reshape(-1, 1024)
        return x



class EncoderRNN(nn.Module):
    def __init__(self, cfg):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(cfg.hidden_size, cfg.hidden_size, cfg.enc_num_layers, batch_first=True)
        
    def forward(self, features):
        """Decode image feature vectors and generates captions."""
        packed = pad_sequence(features, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        return hiddens[:, -1, :] # last hidden state of each sequence


class DecoderRNN(nn.Module):
    def __init__(self, cfg):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.hidden_size, cfg.hidden_size, cfg.dec_num_layers, batch_first=True)
        self.max_seg_length = cfg.max_seq_length
        self.linear = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        
    def forward(self, features):
        """Decode image feature vectors and generates captions."""
        features = features.unsqueeze(1).permute(1, 0, 2) # (batch_size, 1, hidden_size)
        rnn_input = torch.zeros(features.shape[1], self.max_seg_length, self.cfg.hidden_size).to(features.device) # (batch_size, max_seq_length, hidden_size)
        # rnn_input[:, 0, :] = features.squeeze(1) # (batch_size, hidden_size)
        c_0 = torch.zeros(features.shape).to(features.device) # (batch_size, hidden_size)
        packed = pad_sequence(rnn_input, batch_first=True) 
        # print(c_0.shape)
        hiddens, _ = self.lstm(packed, (features, c_0))
        outputs = self.linear(hiddens)
        return outputs



# MART model
class RecursiveTransformer(nn.Module):
    def __init__(self, cfg: MartConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn_enc = RegressionNet()
        self.rnn_enc = EncoderRNN(cfg)
        self.rnn_dec = DecoderRNN(cfg)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    def forward_step(
        self, video_features
    ):
        """
        single step forward in the recursive structure
        """
        hidden_features = []
        for i in range(self.cfg.num_img):
            hidden_features.append(self.cnn_enc(video_features[:, i, :, :, :].squeeze()))
        hidden_features = torch.stack(hidden_features).permute(1, 0, 2)
        hidden_features = self.rnn_enc(hidden_features)
        hidden_features = self.rnn_dec(hidden_features)
        # print(hidden_features.shape)
        return hidden_features

    # ver. future
    def forward(
        self,
        input_ids_list,
        video_features_list
    ):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits,
                `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions,
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,

        Returns:
        """
        # [(N, M, D)] * num_hidden_layers, initialized internally
        step_size = len(input_ids_list)
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        hidden_features = self.forward_step(
            video_features_list[0]
        )
        prediction_scores_list.append(hidden_features)
        # compute loss, get predicted words
        caption_loss = 0.0
        # for idx in range(step_size):
        snt_loss = self.loss_func(
            prediction_scores_list[0].permute(0, 2, 1),
            input_ids_list[0],
        )
        caption_loss += snt_loss
        caption_loss /= step_size
        return caption_loss, prediction_scores_list

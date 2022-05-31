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
from torch.nn.utils import weight_norm
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


class FeatureExtractor(torch.nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # self.conv1 = torch.nn.Conv3d(3, 256, (1, 8, 8), stride=(1, 2, 2))
        # self.conv2 = torch.nn.Conv3d(256, 512, (1, 8, 8), stride=(1, 2, 2))
        # self.conv3 = torch.nn.Conv3d(512, 1024, (1, 8, 8), stride=(1, 2, 2))
        # self.conv4 = torch.nn.Conv3d(1024, 768, (1, 8, 8), stride=(1, 2, 2))
        # self.conv5 = torch.nn.Conv3d(768, 512, (1, 8, 8), stride=(1, 1, 1))
        # self.conv6 = torch.nn.Conv3d(512, 512, (1, 1, 1), stride=(1, 1, 1))
        # self.conv7 = torch.nn.Conv3d(512, 512, (1, 1, 1), stride=(1, 1, 1))
        self.conv1 = torch.nn.Conv3d(3, 256, (1, 8, 8), stride=(1, 2, 2))
        self.conv2 = torch.nn.Conv3d(256, 512, (1, 8, 8), stride=(1, 2, 2))
        self.conv3 = torch.nn.Conv3d(512, 1024, (1, 8, 8), stride=(1, 1, 1))
        self.conv4 = torch.nn.Conv3d(1024, 768, (1, 8, 8), stride=(1, 2, 2))
        self.conv5 = torch.nn.Conv3d(768, 512, (1, 7, 7), stride=(1, 1, 1))
        self.conv6 = torch.nn.Conv3d(512, 512, (1, 1, 1), stride=(1, 1, 1))
        self.conv7 = torch.nn.Conv3d(512, 512, (1, 1, 1), stride=(1, 1, 1))


    def forward(self, x): # input: (batch_size, 5, 224, 224, 3)
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, 3, 5, 224, 224)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x).reshape(-1, 5, 512)
        return x

class TemporalConvNet(nn.Module):
    def __init__(self, config):
        super(TemporalConvNet, self).__init__()
        # self.conv1 = nn.Conv1d(512, 768, 1)
        # self.conv2 = nn.Conv1d(768, 1024, 1)
        # self.conv3 = nn.Conv1d(1024, 768, 1)
        # self.conv4 = nn.Conv1d(768, 512, 1)
        # self.conv5 = nn.Conv1d(512, 512, 1)
        self.conv1 = nn.Conv1d(512, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        # self.conv3 = nn.Conv1d(1024, 768, 1)
        # self.conv4 = nn.Conv1d(768, 512, 1)
        # self.conv5 = nn.Conv1d(512, 512, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # input: (batch_size, seq_len, num_features)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.conv5(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.conv5(x)
        x = x.permute(0, 2, 1) # output: (batch_size, num_features, seq_len)
        return x

class FeaturePredictor(torch.nn.Module):
    def __init__(self, config):
        super(FeaturePredictor, self).__init__()
        self.cfg = config
        self.tcn = TemporalConvNet(config)

    def forward(self, x):
        hidden = self.tcn(x)
        x = x + hidden
        return x


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()
        self.in_projection = nn.Linear(conv_channels, embed_dim)
        self.out_projection = nn.Linear(embed_dim, conv_channels)
        self.bmm = torch.bmm

    def forward(self, x, wordemb, imgsfeats):
        residual = x
        x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)
        b, c, f_h, f_w = imgsfeats.size()
        y = imgsfeats.view(b, c, f_h*f_w).permute(0, 2, 1)
        x = self.bmm(x, y)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
        x = x.view(sz)
        attn_scores = x
        y = y.permute(0, 2, 1)
        x = self.bmm(x, y)
        s = y.size(1)
        x = x * (s * math.sqrt(1.0 / s))
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

class convcap(nn.Module):
    def __init__(self, config, num_layers=3, is_attention=True, nfeats=300, dropout=0.1):
        super(convcap, self).__init__()
        self.cfg = config
        num_wordclass = self.cfg.vocab_size
        self.nimgfeats = 512
        self.is_attention = is_attention
        self.nfeats = nfeats
        self.dropout = dropout 
        self.emb_0 = nn.Linear(nfeats, nfeats)
        self.dropout_0 = nn.Dropout(0.1)
        self.emb_1 = nn.Linear(nfeats, nfeats)
        self.imgproj = nn.Linear(self.nimgfeats, self.nfeats)
        self.resproj = nn.Linear(nfeats*2, self.nfeats)
        n_in = 2 * self.nfeats 
        n_out = self.nfeats
        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = 5
        self.pad = self.kernel_size - 1
        for i in range(self.n_layers):
            self.convs.append(nn.Conv1d(n_in, 2*n_out, self.kernel_size, padding=int(self.pad)))
            if(self.is_attention):
                self.attention.append(AttentionLayer(n_out, nfeats))
            n_in = n_out
        self.classifier_0 = nn.Linear(self.nfeats, (nfeats // 2))
        self.classifier_1 = nn.Linear((nfeats // 2), num_wordclass)

    def forward(self, imgsfeats, wordclass):
        attn_buffer = None
        wordemb = self.dropout_0(F.relu(self.emb_0(wordclass)))
        wordemb = self.emb_1(wordemb)
        x = wordemb.transpose(2, 1)
        batchsize, wordembdim, maxtokens = x.size() # (16, 300, 25)
        y = F.relu(self.imgproj(imgsfeats))
        # y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
        imgsfeats = y.unsqueeze(2)
        y = y.squeeze().permute(0, 2, 1).repeat(1, 1, 5)
        x = torch.cat([x, y], 1)
        for i, conv in enumerate(self.convs):
            if(i == 0):
                x = x.transpose(2, 1)
                residual = self.resproj(x)
                residual = residual.transpose(2, 1)
                x = x.transpose(2, 1)
            else:
                residual = x
            x = F.dropout(x, 0.1, training=self.training)
            x = conv(x)
            x = x[:,:,:-self.pad]
            x = F.glu(x, dim=1)
            if(self.is_attention):
                attn = self.attention[i]
                x = x.transpose(2, 1)
                x, attn_buffer = attn(x, wordemb, imgsfeats)
                x = x.transpose(2, 1)
            x = (x+residual)*math.sqrt(.5)
        x = x.transpose(2, 1)
        x = self.classifier_0(x)
        x = F.dropout(x, 0.1, training=self.training)
        x = self.classifier_1(x)
        x = x.transpose(2, 1)
        return x, attn_buffer

class CaptioningModule(nn.Module):
    def __init__(self, cfg):
        """Set the hyper-parameters and build the layers."""
        super(CaptioningModule, self).__init__()
        self.cfg = cfg
        self.conv = convcap(cfg)

    def forward(self, imgsfeats):
        """Run forward propagation."""
        # print(imgsfeats[0].size())
        wordclass = torch.zeros((imgsfeats.size(0), self.cfg.max_seq_length, self.cfg.word_feat_size), requires_grad=True).cuda()
        outputs, attn = self.conv(imgsfeats, wordclass)
        return outputs



# MART model
class RecursiveTransformer(nn.Module):
    def __init__(self, cfg: MartConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = FeatureExtractor()
        self.enc_comv = FeaturePredictor(self.cfg)
        self.dec_conv = CaptioningModule(self.cfg)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.l1 = nn.L1Loss()

    def forward_step(
        self, video_features, gt = None
    ):
        """
        single step forward in the recursive structure
        """
        video_features = self.embedding(video_features)
        hidden_features = self.enc_comv(video_features)
        outputs = self.dec_conv(hidden_features)
        if gt is not None:
            gt = self.embedding(gt)
            return outputs, video_features, gt
        # print(hidden_features.shape)
        return outputs, video_features, gt

    # ver. future
    def forward(
        self,
        input_ids_list,
        video_features_list,
        gt_clip = None
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
        predict_feats = []
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        outputs, hidden_features, gt = self.forward_step(
            video_features_list[0], gt_clip[0]
        )
        prediction_scores_list.append(outputs)
        predict_feats.append(hidden_features)
        # compute loss, get predicted words
        caption_loss = 0.0
        # for idx in range(step_size):
        snt_loss = self.loss_func(
            prediction_scores_list[0],
            input_ids_list[0],
        )
        l1 = 0.0
        if gt is not None:
            l1 = self.l1(hidden_features, gt)
        caption_loss += snt_loss + l1
        caption_loss /= step_size
        return caption_loss, prediction_scores_list

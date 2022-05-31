"""
Trainer for retrieval training and validation. Holds the main training loop.
"""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union
import sys

import numpy as np
import torch as th
from torch import nn
from torch.cuda.amp import autocast
from torch.utils import data
from tqdm import tqdm

from coot.configs_retrieval import ExperimentTypesConst
from mart.caption_eval_tools import get_reference_files
from mart.configs_mart import MartConfig, MartMetersConst as MMeters
from mart.evaluate_language import evaluate_language_files
from mart.evaluate_repetition import evaluate_repetition_files
from mart.evaluate_stats import evaluate_stats_files
from mart.optimization import BertAdam, EMA
from mart.recursive_caption_dataset import RecursiveCaptionDataset, prepare_batch_inputs
from mart.translator import Translator
from nntrainer import trainer_base
from nntrainer.experiment_organization import ExperimentFilesHandler
from nntrainer.metric import (
    TRANSLATION_METRICS,
    TextMetricsConst,
    TextMetricsConstEvalCap,
)
from nntrainer.models import BaseModelManager
from nntrainer.trainer_configs import BaseTrainerState
from nntrainer.utils import TrainerPathConst
import wandb


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


# only log the important ones to console
TRANSLATION_METRICS_LOG = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "re4"]


class MartFilesHandler(ExperimentFilesHandler):
    """
    Overwrite default filehandler to add some more paths.
    """

    def __init__(
        self,
        exp_group: str,
        exp_name: str,
        run_name: str,
        log_dir: str = TrainerPathConst.DIR_EXPERIMENTS,
        annotations_dir: str = TrainerPathConst.DIR_ANNOTATIONS,
    ):
        super().__init__(
            ExperimentTypesConst.CAPTION, exp_group, exp_name, run_name, log_dir=log_dir
        )
        self.annotations_dir = annotations_dir
        self.path_caption = self.path_base / TrainerPathConst.DIR_CAPTION

    def get_translation_files(self, epoch: Union[int, str], split: str) -> Path:
        """
        Get all file paths for storing translation results and evaluation.
        Args:
            epoch: Epoch.
            split: dataset split (val, test)
        Returns:
            Path to store raw model output and ground truth.
        """
        return (
            self.path_caption
            / f"{TrainerPathConst.FILE_PREFIX_TRANSL_RAW}_{epoch}_{split}.json"
        )

    def setup_dirs(self, *, reset: bool = False) -> None:
        """
        Call super class to setup directories and additionally create the caption folder.
        Args:
            reset:
        Returns:
        """
        super().setup_dirs(reset=reset)
        os.makedirs(self.path_caption, exist_ok=True)


class MartModelManager(BaseModelManager):
    """
    Wrapper for MART models.
    """

    def __init__(self, cfg: MartConfig, model: nn.Module):
        super().__init__(cfg)
        # update config type hints
        self.cfg: MartConfig = self.cfg
        self.model_dict: [str, nn.Module] = {"model": model}


class MartTrainerState(BaseTrainerState):
    prev_best_score = 0.0
    es_cnt = 0


class MartTrainer(trainer_base.BaseTrainer):
    """
    Trainer for retrieval.
    Notes:
        The parent TrainerBase takes care of all the basic stuff: Setting up directories and logging,
        determining device and moving models to cuda, setting up checkpoint loading and metrics.
    Args:
        cfg: Loaded configuration instance.
        model: Model.
        exp_group: Experiment group.
        exp_name: Experiment name.
        run_name: Experiment run.
        train_loader_length: Length of the train loader, required for some LR schedulers.
        log_dir: Directory to put results.
        log_level: Log level. None will default to INFO = 20 if a new logger is created.
        logger: Logger. With the default None, it will be created by the trainer.
        print_graph: Print graph and forward pass of the model.
        reset: Delete entire experiment and restart from scratch.
        load_best: Whether to load the best epoch (default loads last epoch to continue training).
        load_epoch: Whether to load a specific epoch.
        load_model: Load model given by file path.
        inference_only: Removes some parts that are not needed during inference for speedup.
        annotations_dir: Folder with ground truth captions.
    """

    def __init__(
        self,
        cfg: MartConfig,
        model: nn.Module,
        exp_group: str,
        exp_name: str,
        run_name: str,
        train_loader_length: int,
        *,
        log_dir: str = "experiments",
        log_level: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        print_graph: bool = False,
        reset: bool = False,
        load_best: bool = False,
        load_epoch: Optional[int] = None,
        load_model: Optional[str] = None,
        inference_only: bool = False,
        annotations_dir: str = TrainerPathConst.DIR_ANNOTATIONS,
    ):
        # create a wrapper for the model
        model_mgr = MartModelManager(cfg, model)

        # overwrite default experiment files handler
        exp = MartFilesHandler(
            exp_group,
            exp_name,
            run_name,
            log_dir=log_dir,
            annotations_dir=annotations_dir,
        )
        exp.setup_dirs(reset=reset)

        super().__init__(
            cfg,
            model_mgr,
            exp_group,
            exp_name,
            run_name,
            train_loader_length,
            ExperimentTypesConst.CAPTION,
            log_dir=log_dir,
            log_level=log_level,
            logger=logger,
            print_graph=print_graph,
            reset=reset,
            load_best=load_best,
            load_epoch=load_epoch,
            load_model=load_model,
            is_test=inference_only,
            exp_files_handler=exp,
        )
        self.model = model
        # ---------- setup ----------

        # update type hints from base classes to inherited classes
        self.cfg: MartConfig = self.cfg
        self.model_mgr: MartModelManager = self.model_mgr
        self.exp: MartFilesHandler = self.exp

        # # overwrite default state with inherited trainer state in case we need additional state fields
        # self.state = RetrievalTrainerState()

        # ---------- loss ----------

        # loss is created directly in the mart model and not needed here

        # ---------- additional metrics ----------
        # train loss and accuracy
        self.metrics.add_meter(MMeters.TRAIN_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.TRAIN_ACC, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_ACC, use_avg=False)

        # track gradient clipping manually
        self.metrics.add_meter(MMeters.GRAD, per_step=True, reset_avg_each_epoch=True)

        # translation metrics (bleu etc.)
        for meter_name in TRANSLATION_METRICS.values():
            self.metrics.add_meter(meter_name, use_avg=False)

        # ---------- optimization ----------

        self.optimizer = None
        self.lr_scheduler = None
        self.ema = EMA(cfg.ema_decay)
        self.best_epoch = 0
        # skip optimizer if not training
        if not self.is_test:
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if cfg.ema_decay > 0:
                # register EMA params
                self.logger.info(
                    f"Registering {sum(p.numel() for p in model.parameters())} params for EMA"
                )
                all_names = []
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.ema.register(name, p.data)
                    all_names.append(name)
                self.logger.debug("\n".join(all_names))

            num_train_optimization_steps = train_loader_length * cfg.train.num_epochs
            self.optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=cfg.lr,
                warmup=cfg.lr_warmup_proportion,
                t_total=num_train_optimization_steps,
                schedule="warmup_linear",
            )

        # ---------- Translator ----------

        self.translator = Translator(self.model, self.cfg, logger=self.logger)

        # post init hook for checkpoint loading
        self.hook_post_init()

        if self.load and not self.load_model:
            # reload EMA weights from checkpoint (the shadow) and save the model parameters (the original)
            ema_file = self.exp.get_models_file_ema(self.load_ep)
            self.logger.info(f"Update EMA from {ema_file}")
            self.ema.set_state_dict(th.load(str(ema_file)))
            self.ema.assign(self.model, update_model=False)

        # disable ema when loading model directly or when decay is 0 / -1
        if self.load_model or cfg.ema_decay <= 0:
            self.ema = None

        self.train_steps = 0
        self.val_steps = 0
        self.test_steps = 0
        self.beforeloss = 0.0
        self.wandb_flag = 0

    def train_model(
        self, train_loader: data.DataLoader, val_loader: data.DataLoader, test_loader
    ) -> None:
        """
        Train epochs until done.
        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        """
        while(True):
            self.wandb_flag = int(input("Use wandb\n Yes: 1, No: 0\n"))
            if self.wandb_flag == 1:
                wandb_name = input("please input project name : ")
                wandb.init(name=wandb_name, project="mart")
                break
            elif self.wandb_flag == 0:
                break
            else:
                continue
        self.hook_pre_train()  # pre-training hook: time book-keeping etc.
        self.steps_per_epoch = len(train_loader)  # save length of epoch

        # ---------- Epoch Loop ----------
        for _epoch in tqdm(range(self.state.current_epoch, self.cfg.train.num_epochs)):
            # if self.check_early_stop():
            #     break
            self.hook_pre_train_epoch()  # pre-epoch hook: set models to train, time book-keeping

            # check exponential moving average
            if (
                self.ema is not None
                and self.state.current_epoch != 0
                and self.cfg.ema_decay != -1
            ):
                # use normal parameters for training, not EMA model
                self.ema.resume(self.model)
            # summary(self.model, [16, 25, 768])

            th.autograd.set_detect_anomaly(True)

            total_loss = 0
            n_word_total = 0
            n_word_correct = 0
            num_steps = 0
            batch_loss = 0.0

            # ---------- Data　loader Iteration ----------
            for step, batch in enumerate(tqdm(train_loader)):
                self.hook_pre_step_timer()  # hook for step timing

                # ---------- forward pass ----------
                self.optimizer.zero_grad()
                with autocast(enabled=self.cfg.fp16_train):
                        # ---------- training step for recurrent models ----------
                    batched_data = [
                        prepare_batch_inputs(
                            step_data,
                            use_cuda=self.cfg.use_cuda,
                            non_blocking=self.cfg.cuda_non_blocking,
                        )
                        for step_data in batch[0]
                    ]

                    input_ids_list = [e["input_ids"] for e in batched_data]
                    video_features_list = [e["video_feature"] for e in batched_data]
                    gtfeats = [e["gt"] for e in batched_data]
                    if self.cfg.debug:
                        cur_data = batched_data[step]
                        self.logger.info(
                            "input_ids \n{}".format(cur_data["input_ids"][step])
                        )
                    # ver. future
                    loss, pred_scores_list = self.model(
                        input_ids_list,
                        video_features_list,
                        gtfeats
                    )
                    self.train_steps += 1
                    num_steps += 1
                    batch_loss += loss

                self.hook_post_forward_step_timer()  # hook for step timing

                # ---------- backward pass ----------
                grad_norm = None
                if self.cfg.fp16_train:
                    # with fp16 amp
                    self.grad_scaler.scale(loss).backward()
                    if self.cfg.train.clip_gradient != -1:
                        # gradient clipping
                        self.grad_scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.train.clip_gradient
                        )
                    # gradient scaler realizes if gradients have been unscaled already and doesn't do it again.
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # with regular float32
                    loss.backward()
                    if self.cfg.train.clip_gradient != -1:
                        # gradient clipping
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.train.clip_gradient
                        )
                    self.optimizer.step()
                # update model parameters with ema
                if self.ema is not None:
                    self.ema(self.model, self.state.total_step)

                # keep track of loss, accuracy, gradient norm
                total_loss += loss.item()
                n_correct = 0
                n_word = 0
                # for pred, gold in zip(pred_scores_list, input_labels_list):
                #     n_correct += cal_performance(pred, gold)
                #     valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
                #     n_word += valid_label_mask.sum().item()
                # n_word_total += n_word
                # n_word_correct += n_correct
                if grad_norm is not None:
                    self.metrics.update_meter(MMeters.GRAD, grad_norm)

                if self.cfg.debug:
                    break

                additional_log = f" Grad {self.metrics.meters[MMeters.GRAD].avg:.2f}"
                self.hook_post_backward_step_timer()  # hook for step timing

                # post-step hook: gradient clipping, profile gpu, update metrics, count step, step LR scheduler, log
                current_lr = self.optimizer.get_lr()[0]
                self.hook_post_step(
                    step,
                    loss,
                    current_lr,
                    additional_log=additional_log,
                    disable_grad_clip=True,
                )

            # log train statistics
            # loss_per_word = 1.0 * total_loss / n_word_total
            # accuracy = 1.0 * n_word_correct / n_word_total
            # self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
            # self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
            # return loss_per_word, accuracy
            batch_loss /= num_steps
            if self.wandb_flag == 1:
                wandb.log({"train_loss": batch_loss})

            # ---------- validation ----------
            do_val = self.check_is_val_epoch()

            is_best = False
            if do_val:
                # run validation including with ground truth tokens and translation without any text
                _val_loss, _val_score, is_best, _metrics = self.validate_epoch(
                    val_loader
                )
                # if is_best:
                print("#############################################")
                print("Do test")
                self.test_epoch(test_loader)
                print("###################################################")

            # save the EMA weights
            ema_file = self.exp.get_models_file_ema(self.state.current_epoch)
            th.save(self.ema.state_dict(), str(ema_file))

            # post-epoch hook: scheduler, save checkpoint, time bookkeeping, feed tensorboard
            self.hook_post_train_and_val_epoch(do_val, is_best)

        # show end of training log message
        self.hook_post_train()
        print("###################################################")
        self.logger.info(
            ", ".join(
                [f"{name} {self.higest_test[name]:.2%}" for name in self.test_metrics]
            )
        )


    @th.no_grad()
    def validate_epoch(
        self, data_loader: data.DataLoader
    ) -> (Tuple[float, float, bool, Dict[str, float]]):
        """
        Run both validation and translation.
        Validation: The same setting as training, where ground-truth word x_{t-1} is used to predict next word x_{t},
        not realistic for real inference.
        Translation: Use greedy generated words to predicted next words, the true inference situation.
        eval_mode can only be set to `val` here, as setting to `test` is cheating
        0. run inference, 1. Get METEOR, BLEU1-4, CIDEr scores, 2. Get vocab size, sentence length
        Args:
            data_loader: Dataloader for validation
        Returns:
            Tuple of:
                validation loss
                validation score
                epoch is best
                custom metrics with translation results dictionary
        """
        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        batch_loss = 0.0
        batch_idx = 0

        # setup ema
        if self.ema is not None:
            self.ema.assign(self.model)

        # setup translation submission
        batch_res = {
            "version": "VERSION 1.0",
            "results": defaultdict(list),
            "external_data": {"used": "true", "details": "ay"},
        }
        dataset: RecursiveCaptionDataset = data_loader.dataset

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        pbar = tqdm(
            total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}"
        )
        for _step, batch in enumerate(data_loader):
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing

            with autocast(enabled=self.cfg.fp16_val):
                if self.cfg.recurrent:
                    # recurrent MART, TransformerXL, ...
                    # get data
                    batched_data = [
                        prepare_batch_inputs(
                            step_data,
                            use_cuda=self.cfg.use_cuda,
                            non_blocking=self.cfg.cuda_non_blocking,
                        )
                        for step_data in batch[0]
                    ]
                    # validate (ground truth as input for next token)
                    input_ids_list = [e["input_ids"] for e in batched_data]
                    video_features_list = [e["video_feature"] for e in batched_data]
                    gtfeats = [e["gt"] for e in batched_data]
                    # input_labels_list = [e["input_labels"] for e in batched_data]

                    # ver. future
                    loss, pred_scores_list = self.model(
                        input_ids_list,
                        video_features_list,
                        gtfeats
                    )
                    batch_loss += loss
                    batch_idx += 1
                    # translate (no ground truth text)
                    step_sizes = batch[1]  # list(int), len == bsz
                    meta = batch[2]  # list(dict), len == bsz

                    model_inputs = [
                        [e["input_ids"] for e in batched_data],
                        [e["video_feature"] for e in batched_data]
                    ]
                    dec_seq_list = self.translator.translate_batch(
                        model_inputs,
                        use_beam=self.cfg.use_beam,
                        recurrent=True,
                        untied=False,
                        xl=self.cfg.xl,
                    )

                    for example_idx, (step_size, cur_meta) in enumerate(
                        zip(step_sizes, meta)
                    ):
                        # print(cur_meta)
                        # example_idx indicates which example is in the batch
                        for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                            # step_idx or we can also call it sen_idx
                            batch_res["results"][cur_meta["clip_id"]].append(
                                {
                                    "sentence": dataset.convert_ids_to_sentence(
                                        step_batch[example_idx].cpu().tolist()
                                    ),
                                    # remove encoding
                                    # .encode("ascii", "ignore"),
                                    "gt_sentence": cur_meta["gt_sentence"],
                                    "clip_id": cur_meta["clip_id"]
                                }
                            )
                    # if self.cfg.debug:
                    #     print(
                    #         f"Vid feat {[v.mean().item() for v in video_features_list]}"
                    #     )

                # keep logs
                n_correct = 0
                n_word = 0
                # for pred, gold in zip(pred_scores_list, input_labels_list):
                #     n_correct += cal_performance(pred, gold)
                #     valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
                #     n_word += valid_label_mask.sum().item()

                # calculate metrix
                # n_word_total += n_word
                # n_word_correct += n_correct
                total_loss += loss.item()

            # end of step
            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            if self.cfg.debug:
                break

            pbar.update()
        pbar.close()

        # ---------- validation done ----------
        batch_loss /= batch_idx
        loss_delta = self.beforeloss - batch_loss
        if self.wandb_flag == 1:
            wandb.log({"val_loss_diff": loss_delta})
            wandb.log({"val_loss": batch_loss})
        self.beforeloss = batch_loss

        # sort translation
        batch_res["results"] = self.translator.sort_res(batch_res["results"])

        # write translation results of this epoch to file
        eval_mode = self.cfg.dataset_val.split  # which dataset split
        file_translation_raw = self.exp.get_translation_files(
            self.state.current_epoch, eval_mode
        )
        json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"))

        # get reference files (ground truth captions)
        reference_files_map = get_reference_files(
            self.cfg.dataset_val.name, self.exp.annotations_dir
        )
        reference_files = reference_files_map[eval_mode]
        reference_file_single = reference_files[0]

        # language evaluation
        res_lang = evaluate_language_files(
            file_translation_raw, reference_files, verbose=False, all_scorer=True
        )
        # basic stats
        res_stats = evaluate_stats_files(
            file_translation_raw, reference_file_single, verbose=False
        )
        # repetition
        res_rep = evaluate_repetition_files(
            file_translation_raw, reference_file_single, verbose=False
        )

        # merge results
        all_metrics = {**res_lang, **res_stats, **res_rep}
        assert len(all_metrics) == len(res_lang) + len(res_stats) + len(
            res_rep
        ), "Lost infos while merging translation results!"

        # flatten results and make them json compatible
        flat_metrics = {}
        for key, val in all_metrics.items():
            if isinstance(val, Mapping):
                for subkey, subval in val.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
                continue
            flat_metrics[key] = val
        for key, val in flat_metrics.items():
            if isinstance(val, (np.float16, np.float32, np.float64)):
                flat_metrics[key] = float(val)

        # feed meters
        for result_key, meter_name in TRANSLATION_METRICS.items():
            self.metrics.update_meter(meter_name, flat_metrics[result_key])

        # log translation results
        self.logger.info(
            f"Done with translation, epoch {self.state.current_epoch} split {eval_mode}"
        )
        self.logger.info(
            ", ".join(
                [f"{name} {flat_metrics[name]:.2%}" for name in TRANSLATION_METRICS_LOG]
            )
        )

        # calculate and output validation metrics
        # loss_per_word = 1.0 * total_loss / n_word_total
        # accuracy = 1.0 * n_word_correct / n_word_total
        # self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
        # self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
        forward_time_total /= num_steps
        # self.logger.info(
        #     f"Loss {loss_per_word:.5f} Acc {accuracy:.3%} total {timer() - self.timer_val_epoch:.3f}s, "
        #     f"forward {forward_time_total:.3f}s"
        # )

        # find field which determines whether this is a new best epoch
        if self.wandb_flag == 1:
            wandb.log({"val_BLEU4": flat_metrics["Bleu_4"], "val_METEOR": flat_metrics["METEOR"], "val_ROUGE_L": flat_metrics["ROUGE_L"], "val_CIDEr": flat_metrics["CIDEr"]})
        if self.cfg.val.det_best_field == "cider":
            # val_score = flat_metrics["CIDEr"]
            val_score = -1 * batch_loss
        else:
            raise NotImplementedError(
                f"best field {self.cfg.val.det_best_field} not known"
            )

        # check for a new best epoch and update validation results
        is_best = self.check_is_new_best(val_score)
        if is_best == True:
            self.best_epoch = self.state.current_epoch
        self.hook_post_val_epoch(batch_loss, is_best)

        if self.is_test:
            # for test runs, save the validation results separately to a file
            self.metrics.feed_metrics(
                False, self.state.total_step, self.state.current_epoch
            )
            metrics_file = (
                self.exp.path_base / f"val_ep_{self.state.current_epoch}.json"
            )
            self.metrics.save_epoch_to_file(metrics_file)
            self.logger.info(f"Saved validation results to {metrics_file}")

            # update the meteor metric in the result if it's -999 because java crashed. only in some conditions
            best_ep = self.exp.find_best_epoch()
            self.logger.info(
                f"Dataset split config {self.cfg.dataset_val.split} loaded {self.load_ep} best {best_ep}"
            )
            if (
                self.cfg.dataset_val.split == "val"
                and self.load_ep == best_ep == self.state.current_epoch
            ):
                # load metrics file and write it back with the new meteor IFF meteor is -999
                metrics_file = self.exp.get_metrics_epoch_file(best_ep)
                metrics_data = json.load(metrics_file.open("rt", encoding="utf8"))
                # metrics has stored meteor as a list of tuples (epoch, value). convert to dict, update, convert back.
                meteor_dict = dict(metrics_data[TextMetricsConst.METEOR])
                if ((meteor_dict[best_ep] + 999) ** 2) < 1e-4:
                    meteor_dict[best_ep] = flat_metrics[TextMetricsConstEvalCap.METEOR]
                    metrics_data[TextMetricsConst.METEOR] = list(meteor_dict.items())
                    json.dump(metrics_data, metrics_file.open("wt", encoding="utf8"))
                    self.logger.info(f"Updated meteor in file {metrics_file}")

        return total_loss, val_score, is_best, flat_metrics


    @th.no_grad()
    def test_epoch(
        self, data_loader: data.DataLoader
    ) -> (Tuple[float, float, bool, Dict[str, float]]):
        """
        Run both validation and translation.
        Validation: The same setting as training, where ground-truth word x_{t-1} is used to predict next word x_{t},
        not realistic for real inference.
        Translation: Use greedy generated words to predicted next words, the true inference situation.
        eval_mode can only be set to `val` here, as setting to `test` is cheating
        0. run inference, 1. Get METEOR, BLEU1-4, CIDEr scores, 2. Get vocab size, sentence length
        Args:
            data_loader: Dataloader for validation
        Returns:
            Tuple of:
                validation loss
                validation score
                epoch is best
                custom metrics with translation results dictionary
        """
        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        # setup ema
        if self.ema is not None:
            self.ema.assign(self.model)

        # setup translation submission
        batch_res = {
            "version": "VERSION 1.0",
            "results": defaultdict(list),
            "external_data": {"used": "true", "details": "ay"},
        }
        dataset: RecursiveCaptionDataset = data_loader.dataset

        # ---------- Dataloader Iteration ----------
        num_steps = 0
        pbar = tqdm(
            total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}"
        )
        batch_loss = 0.0
        batch_idx = 0
        for _step, batch in enumerate(data_loader):
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing

            with autocast(enabled=self.cfg.fp16_val):
                if self.cfg.recurrent:
                    # recurrent MART, TransformerXL, ...
                    # get data
                    batched_data = [
                        prepare_batch_inputs(
                            step_data,
                            use_cuda=self.cfg.use_cuda,
                            non_blocking=self.cfg.cuda_non_blocking,
                        )
                        for step_data in batch[0]
                    ]
                    # validate (ground truth as input for next token)
                    input_ids_list = [e["input_ids"] for e in batched_data]
                    video_features_list = [e["video_feature"] for e in batched_data]
                    gtfeats = [e["gt"] for e in batched_data]
                    # input_labels_list = [e["input_labels_list"] for e in batched_data]
                    # ver. future
                    loss, pred_scores_list = self.model(
                        input_ids_list,
                        video_features_list,
                        gtfeats
                    )
                    batch_loss += loss
                    batch_idx += 1
                    # translate (no ground truth text)
                    step_sizes = batch[1]  # list(int), len == bsz
                    meta = batch[2]  # list(dict), len == bsz

                    model_inputs = [
                        [e["input_ids"] for e in batched_data],
                        [e["video_feature"] for e in batched_data],
                    ]
                    dec_seq_list = self.translator.translate_batch(
                        model_inputs,
                        use_beam=self.cfg.use_beam,
                        recurrent=True,
                        untied=False,
                        xl=self.cfg.xl,
                    )

                    for example_idx, (step_size, cur_meta) in enumerate(
                        zip(step_sizes, meta)
                    ):
                        # example_idx indicates which example is in the batch
                        for step_idx, step_batch in enumerate(dec_seq_list[:step_size]):
                            # step_idx or we can also call it sen_idx
                            batch_res["results"][cur_meta["clip_id"]].append(
                                {
                                    "sentence": dataset.convert_ids_to_sentence(
                                        step_batch[example_idx].cpu().tolist()
                                    ),
                                    "gt_sentence": cur_meta["gt_sentence"],
                                    "clip_id": cur_meta["clip_id"]
                                }
                            )

                # keep logs
                n_correct = 0
                n_word = 0
                # for pred, gold in zip(pred_scores_list, input_labels_list):
                #     n_correct += cal_performance(pred, gold)
                #     valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
                #     n_word += valid_label_mask.sum().item()

                # calculate metrix
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

            # end of step
            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            if self.cfg.debug:
                break

            pbar.update()
        pbar.close()
        batch_loss /= batch_idx
        if self.wandb_flag == 1:
            wandb.log({"test_loss": batch_loss})

        # ---------- validation done ----------

        # sort translation
        batch_res["results"] = self.translator.sort_res(batch_res["results"])

        # write translation results of this epoch to file
        eval_mode = "test"  # which dataset split
        file_translation_raw = self.exp.get_translation_files(
            self.state.current_epoch, "test"
        )
        json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"))

        # get reference files (ground truth captions)
        reference_files_map = get_reference_files(
            self.cfg.dataset_val.name, self.exp.annotations_dir, test=True
        )
        reference_files = reference_files_map[eval_mode]
        reference_file_single = reference_files[0]

        # language evaluation
        res_lang = evaluate_language_files(
            file_translation_raw, reference_files, verbose=False, all_scorer=True
        )
        # basic stats
        res_stats = evaluate_stats_files(
            file_translation_raw, reference_file_single, verbose=False
        )
        # repetition
        res_rep = evaluate_repetition_files(
            file_translation_raw, reference_file_single, verbose=False
        )

        # merge results
        all_metrics = {**res_lang, **res_stats, **res_rep}
        assert len(all_metrics) == len(res_lang) + len(res_stats) + len(
            res_rep
        ), "Lost infos while merging translation results!"

        # flatten results and make them json compatible
        flat_metrics = {}
        for key, val in all_metrics.items():
            if isinstance(val, Mapping):
                for subkey, subval in val.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
                continue
            flat_metrics[key] = val
        for key, val in flat_metrics.items():
            if isinstance(val, (np.float16, np.float32, np.float64)):
                flat_metrics[key] = float(val)

        # feed meters
        for result_key, meter_name in TRANSLATION_METRICS.items():
            self.metrics.update_meter(meter_name, flat_metrics[result_key])

        # log translation results
        self.logger.info(
            f"Done with translation, epoch {self.state.current_epoch} split {eval_mode}"
        )
        if self.wandb_flag == 1:
            wandb.log({"test_BLEU4": flat_metrics["Bleu_4"], "test_METEOR": flat_metrics["METEOR"], "test_ROUGE_L": flat_metrics["ROUGE_L"], "test_CIDEr": flat_metrics["CIDEr"]})
        self.test_metrics = TRANSLATION_METRICS_LOG
        self.higest_test = flat_metrics
        self.logger.info(
            ", ".join(
                [f"{name} {flat_metrics[name]:.2%}" for name in TRANSLATION_METRICS_LOG]
            )
        )


    def get_opt_state(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Return the current optimizer and scheduler state.
        Note that the BertAdam optimizer used already includes scheduling.
        Returns:
            Dictionary of optimizer and scheduler state dict.
        """
        return {
            "optimizer": self.optimizer.state_dict()
            # "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_opt_state(self, opt_state: Dict[str, Dict[str, nn.Parameter]]) -> None:
        """
        Set the current optimizer and scheduler state from the given state.
        Args:
            opt_state: Dictionary of optimizer and scheduler state dict.
        """
        self.optimizer.load_state_dict(opt_state["optimizer"])
        # self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])

    def get_files_for_cleanup(self, epoch: int) -> List[Path]:
        """
        Implement this in the child trainer.
        Returns:
            List of files to cleanup.
        """
        return [
            # self.exp.get_translation_files(epoch, split="train"),
            self.exp.get_translation_files(epoch, split="val"),
            self.exp.get_models_file_ema(epoch),
        ]
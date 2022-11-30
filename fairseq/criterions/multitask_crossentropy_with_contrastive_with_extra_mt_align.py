# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.criterions.label_smoothed_cross_entropy_with_contrastive_loss import \
    LabelSmoothedCrossEntropyWithContrastiveCriterion
import torch.nn.functional as F


@register_criterion("multi_task_cross_entropy_with_contrastive_with_extra_MT_align")
class MultiTaskCrossEntropyWithContrastiveWithExtraMT(LabelSmoothedCrossEntropyWithContrastiveCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            contrastive_weight=0.0,
            contrastive_temperature=1.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy,
                         contrastive_weight, contrastive_temperature)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_asr, nll_loss_asr = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_mt, nll_loss_mt = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_tag, nll_loss_tag = torch.tensor(0.0), torch.tensor(0.0)
        contrastive_loss, short_audio_len = torch.tensor(0.0), None
        loss = torch.zeros(1, requires_grad=True)

        _net_output = model(**sample["net_input"])  # (x, extra)
        target = sample["tag"]
        _,logits = _net_output
        # print(logits.size())
        # print(target.size())
        # print()
        # if self.ignore_prefix_size > 0:
        #     logits = logits[self.ignore_prefix_size:, :, :].contiguous()
        #     target = target[self.ignore_prefix_size:, :].contiguous()
        # print("**************source******************")
        # print(sample["net_input"]["src_tokens"])
        # print("**************target1******************")
        # print(target)
        # print('**************************')
        # print(sample["nsentences"])
        # print(target.size())
        # print(logits.size())
        
        logits = logits.transpose(0,1).contiguous().float()
        fertilities =  logits.max(-1)[1]
        # print("**************target2******************")
        # print(fertilities)
        # print(fertilities.size())
        # print(target.size())


        # logits = logits.view(-1, logits.size(-1)).to(dtype=torch.float32).contiguous()
        # target = target.view(-1).to(dtype=torch.int64).contiguous()

        fertilities = fertilities.view(-1)
        target = target.view(-1)
        
        # train_x=torch.tensor(train_x,dtype=torch.float32).cuda()
        # train_y=torch.tensor(train_y,dtype=torch.int64).cuda()
        # print("**************target3******************")
        # print(fertilities)
        # print("**************target4******************")
        # print(target)
        # print(fertilities.size())
        # print(target.size())
        pad_mask = target.eq(self.padding_idx)
        pad_mask = ~pad_mask
        # print(fertilities[pad_mask])
        # print(target[pad_mask])
        # print(fertilities[pad_mask])
        # print(target[pad_mask])
        # loss = F.cross_entropy(logits[pad_mask], target[pad_mask], reduction="mean")
        loss = sum(abs(fertilities[pad_mask]-target[pad_mask]))/len(fertilities[pad_mask])
        loss.requires_grad_(True)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        logging_output = {
            "loss": loss.data
        }

        return loss, sample_size, logging_output

    

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        target_ntokens = sum(log.get("target_ntokens", 0) for log in logging_outputs)
        source_ntokens = sum(log.get("source_ntokens", 0) for log in logging_outputs)
        target_ntokens_mt = sum(log.get("target_ntokens_mt", 0) for log in logging_outputs)
        target_ntokens_st = sum(log.get("target_ntokens_st", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        mt_nsentences = sum(log.get("mt_nsentences", 0) for log in logging_outputs)
        st_nsentences = sum(log.get("st_nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        nll_loss_sum_asr = sum(log.get("nll_loss_asr", 0) for log in logging_outputs)
        nll_loss_sum_mt = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)

        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)


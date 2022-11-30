import math
import torch.nn.functional as F
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.criterions.label_smoothed_cross_entropy_with_contrastive_loss import LabelSmoothedCrossEntropyWithContrastiveCriterion

@register_criterion('multi_task_cross_entropy_with_contrastive_with_extra_MT_attn')
class MultiTaskCrossEntropyWithContrastiveWithExtraMT(LabelSmoothedCrossEntropyWithContrastiveCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, 
                report_accuracy=False, contrastive_weight=0.0, attn_weight=0.0, start_kl = 10000, 
                use_tag=False, need_all_layer_attn = False, contrastive_temperature=1.0):

        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, 
        report_accuracy, contrastive_weight, attn_weight, start_kl, use_tag, need_all_layer_attn, contrastive_temperature)

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
        label_smoothed_nll_loss_attn, nll_loss_attn = torch.tensor(0.0), torch.tensor(0.0)
        contrastive_loss, short_audio_len = torch.tensor(0.0), None
        attn_st, attn_mt = torch.tensor(0.0), torch.tensor(0.0)
        loss_kl = torch.tensor(0.0)
        loss_attn_sum = torch.tensor(0.0)
        type_loss = torch.tensor(0.0)
        if "mode" in sample["net_input"] and sample["net_input"]["mode"] == "text_to_text":
            sample["dataset_type"] = "mt"
            sample["net_input"]["is_text_input"] = True
        else:
            sample["net_input"]["is_text_input"] = False

        if model.training and self.need_all_layer_attn:
            # _net_output = model(**sample['net_input'])
            _net_output = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'],
                                    sample["net_input"]["is_text_input"],self.need_all_layer_attn)
        else:
            _net_output = model(**sample['net_input'])

        if model.training:
            net_output, encoder_out = _net_output
            if (sample["dataset_type"] != "mt") and (self.contrastive_weight > 0):
                contrastive_loss, short_audio_len = self.compute_contrastive_loss(
                    model, sample, encoder_out,
                    reduce=reduce, return_short_audio_len=True
                    )
        else:
            net_output = _net_output
            
        if sample['target'] is not None:
            if sample['dataset_type'] == 'st' and model.training:
                label_smoothed_nll_loss_asr, nll_loss_asr, attn_asr,net_output_asr = self.compute_loss_asr(model, sample, reduce=reduce,)
                label_smoothed_nll_loss_mt, nll_loss_mt, attn_mt,net_output_mt = self.compute_loss_mt(model, sample, reduce=reduce)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                # print(self.start_kl)
                if model.num_updates > self.start_kl:
                    # and model.num_updates % 10 > 4
                    # model.num_updates > self.start_kl
                    if not self.need_all_layer_attn:
                        type_loss = label_smoothed_nll_loss.new()
                        loss_kl = label_smoothed_nll_loss.new()
                        loss_attn = label_smoothed_nll_loss.new()                        
                        attn_st = net_output[1]['attn'][0]
                        attn_asr = attn_asr.float()
                        attn_mt = attn_mt.float()
                        # print('********************')
                        trg_mask = sample['target'].eq(self.padding_idx)
                        src_mask = sample['source'].eq(self.padding_idx)
                        audio_mask = encoder_out.encoder_padding_mask
                        attn_mt_mask = attn_mt < 0.05
                        attn_asr_mask = attn_asr < 0.05
                        attn_mt = attn_mt.masked_fill(attn_mt_mask, 0)
                        attn_asr = attn_asr.masked_fill(attn_asr_mask, 0)
                        attn_st_hypo = torch.bmm(attn_mt, attn_asr).detach()
                        values, _ = torch.max(attn_st_hypo,dim=-1)

                        values_mask = values == 0
                        values = values.masked_fill(values_mask, float("-inf"))
                        attn_st_hypo =(attn_st_hypo / values.unsqueeze(-1))*2
                        
                        attn_st_hypo.type_as(attn_st)
                        attn_st_hypo_float = utils.softmax(attn_st_hypo, dim=(-1))
                        attn_st_hypo = attn_st_hypo_float.type_as(attn_st_hypo)
                        attn_st_hypo_detach = attn_st_hypo.detach()
                        # print(audio_mask.size())
                        # print(src_mask.size())
                        # print(trg_mask.size())
                        # print(attn_asr.size())
                        # print(attn_mt.size())
                        # attn_asr = attn_asr.masked_fill(src_mask.unsqueeze(-1), float(0))
                        # attn_asr = attn_asr.masked_fill(audio_mask.unsqueeze(1), float(0))
                        # attn_mt = attn_mt.masked_fill(trg_mask.unsqueeze(-1), float(0))
                        # attn_mt = attn_mt.masked_fill(src_mask.unsqueeze(1), float(0))
                        
                        # #欧式距离loss  hard tag
                        # bsz, tgt_len, src_len_st = attn_st.size()
                        # # attn_hard = torch.zeros(bsz*tgt_len, src_len_st)
                        # target_attn = torch.argmax(attn_st_hypo_detach,dim=-1)
                        # attn_hard = target_attn.new(bsz*tgt_len, src_len_st).fill_(0)
                        # attn_hard = torch.scatter(attn_hard,-1,target_attn.view(-1).contiguous().unsqueeze(-1),1)
                        # attn_hard = attn_hard.view(bsz, tgt_len, src_len_st)
                        # attn_hard.to(attn_st)
                        # attn_hard = attn_hard.masked_fill(trg_mask.unsqueeze(-1), 0)
                        # attn_hard = attn_hard.masked_fill(audio_mask.unsqueeze(1), 0)
                        # attn_st = attn_st.masked_fill(trg_mask.unsqueeze(-1), 0)
                        # attn_st = attn_st.masked_fill(audio_mask.unsqueeze(1), 0)
                        # loss_attn = ( attn_st - attn_hard).norm(dim=2)
                        # # loss_attn = loss_attn.masked_fill(trg_mask, float(0))
                        # loss_attn = loss_attn.sum()


                        #欧式距离loss    
                        attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(trg_mask.unsqueeze(-1), 0)
                        attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(audio_mask.unsqueeze(1), 0)
                        attn_st = attn_st.masked_fill(trg_mask.unsqueeze(-1), 0)
                        attn_st = attn_st.masked_fill(audio_mask.unsqueeze(1), 0)
                        loss_attn = (attn_st_hypo_detach - attn_st).norm(dim=2)
                        # loss_attn = loss_attn.masked_fill(trg_mask, float(0))
                        loss_attn = loss_attn.sum()
                    
                    # # kl loss
                    # attn_st_log = torch.log(attn_st)
                    # attn_st_log = attn_st_log.masked_fill(trg_mask.unsqueeze(-1), 0)
                    # attn_st_log = attn_st_log.masked_fill(audio_mask.unsqueeze(1), 0)
                    # attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(trg_mask.unsqueeze(-1), 0)
                    # attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(audio_mask.unsqueeze(1), 0)
                    # attn_st_hypo_detach = attn_st_hypo_detach.to(torch.float32)
                    # attn_st_log = attn_st_log.to(torch.float32)
                    # loss_kl = F.kl_div(attn_st_log,attn_st_hypo_detach,reduction = 'sum') * 100
                    # loss_kl.type_as(label_smoothed_nll_loss)

                    # nll_loss
                    # attn_st_log = torch.log(attn_st)
                    # attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(trg_mask.unsqueeze(-1), 0)
                    # attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(audio_mask.unsqueeze(1), 0)
                    # target_attn = torch.argmax(attn_st_hypo_detach,dim=-1)
                    # target_attn = target_attn.masked_fill(trg_mask, 0)
                    # # attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(audio_mask.unsqueeze(1), 0)
                    # lprobs_attn = attn_st_log
                    # #nll——loss
                    # label_smoothed_nll_loss_attn, nll_loss_attn = self.compute_loss_attn_no_mfa(model, sample, 
                    #                                                                             lprobs_attn, target_attn, reduce=reduce)
                    # attn_st_hypo_log = utils.log_softmax(attn_st_hypo, dim=(-1))
                    # print(print(torch.max(attn_st_hypo)))
                    # attn_st_hypo = attn_st_hypo.masked_fill_(pad_attn_mask, 0)
                    else:
                        asr_lengths = sample["source_lengths"]
                        mt_lengths = sample["target_lengths"].unsqueeze(-1)
                        audio_lengths = encoder_out.output_encoder_lengths
                        
                        # print(asr_lengths.size())
                        # assert 1==2
                        type_loss = label_smoothed_nll_loss.new()
                        loss_kl = label_smoothed_nll_loss.new()
                        loss_attn = label_smoothed_nll_loss.new()
                        attn_st_all = net_output[1]['all_layer_attn']
                        attn_asr_all = net_output_asr[1]['all_layer_attn']
                        attn_mt_all = net_output_mt[1]['all_layer_attn']
                        # print('********************')
                        trg_mask = sample['target'].eq(self.padding_idx)
                        src_mask = sample['source'].eq(self.padding_idx)
                        audio_mask = encoder_out.encoder_padding_mask
                        bsz, tgt_len, audio_len = attn_st_all[0].size()
                        _, src_len, _ = attn_asr_all[0].size()
                        # for i in range(3,len(attn_st_all)):
                        for i in range(3,len(attn_st_all)):
                            attn_st = attn_st_all[i]
                            attn_mt = attn_mt_all[i].float()
                            attn_asr = attn_asr_all[i].float()

                            reverse_asr_attn = attn_asr.transpose(1,2)
                            reverse_asr_attn_mask = reverse_asr_attn > 0.1
                            reverse_asr_attn_mask2 = reverse_asr_attn > 0.05

                            reverse_asr_attn_mask = torch.sum(reverse_asr_attn_mask,dim=-1)
                            reverse_asr_attn_mask2 = torch.sum(reverse_asr_attn_mask2,dim=-1)
                            reverse_asr_attn_mask = reverse_asr_attn_mask / asr_lengths
                            reverse_asr_attn_mask2 = reverse_asr_attn_mask2 / asr_lengths

                            reverse_asr_attn_mask = reverse_asr_attn_mask >= 0.5
                            src_lenfth_mask = asr_lengths > 15
                            reverse_asr_attn_mask = reverse_asr_attn_mask & src_lenfth_mask
                            reverse_asr_attn_mask2 = reverse_asr_attn_mask2 >= 0.8
                            reverse_asr_attn_mask2 = reverse_asr_attn_mask2 & src_lenfth_mask
                            empty_mask = reverse_asr_attn_mask + reverse_asr_attn_mask2 #判断出是不是空音节
                            # print(reverse_asr_attn.size())
                            reverse_asr_attn = reverse_asr_attn.masked_fill(empty_mask.unsqueeze(-1),0)

                            # attn_asr_hard_idx = torch.argmax(reverse_asr_attn,dim=-1)
                            # attn_asr_hard = attn_asr.new(bsz*audio_len, tgt_len).fill_(0)
                            # attn_asr_hard = torch.scatter(attn_asr_hard,-1,attn_asr_hard_idx.view(-1).contiguous().unsqueeze(-1),1)
                            # attn_asr_hard = attn_asr_hard.view(bsz, tgt_len, audio_len)
                            # attn_asr_hard = attn_asr_hard.masked_fill(empty_mask.unsqueeze(-1),0).transpose(1,2).contiguous()

                            values, idx = torch.sort(reverse_asr_attn,dim = -1)
                            attn_asr_hard = attn_asr.new(bsz*audio_len, src_len).fill_(0)
                            idx = idx[:,:,-2:]
                            no1 = idx[:,:,-1]
                            no2 = idx[:,:,-2]
                            value1 = values[:,:,-1]
                            value2 = values[:,:,-2]
                            sub_value = (value1-value2) <= 0.1
                            mask = abs(no1 - no2)==1
                            mask = mask & sub_value
                            rmask = ~mask
                            no2 = no2.masked_fill(rmask,0)
                            no2 = no1.masked_fill(mask,0) + no2
                            attn_asr_hard = torch.scatter(attn_asr_hard,-1,no2.view(-1).contiguous().unsqueeze(-1),0.5)
                            attn_asr_hard = torch.scatter(attn_asr_hard,-1,no1.view(-1).contiguous().unsqueeze(-1),1).view(bsz, audio_len, src_len)
                            
                            # print(attn_asr_hard.size())
                            attn_asr_hard = attn_asr_hard.masked_fill(empty_mask.unsqueeze(-1),0).transpose(1,2).contiguous()


                            attn_asr = reverse_asr_attn.transpose(1,2)
                            
                            reverse_mt_attn = attn_mt.transpose(1,2)
                            reverse_mt_attn_mask = reverse_mt_attn > 0.1
                            reverse_mt_attn_mask = torch.sum(reverse_mt_attn_mask,dim=-1)
                            reverse_mt_attn_mask = reverse_mt_attn_mask / mt_lengths
                            reverse_mt_attn_mask = reverse_mt_attn_mask >= 0.5
                            mt_lenfth_mask = mt_lengths > 15
                            reverse_mt_attn_mask = reverse_mt_attn_mask & mt_lenfth_mask
                            reverse_mt_attn = reverse_mt_attn.masked_fill(reverse_mt_attn_mask.unsqueeze(-1),0)
                            attn_mt = reverse_mt_attn.transpose(1,2)
                            attn_st_hypo = torch.bmm(attn_mt, attn_asr_hard).detach()

                            # eos_idx = (mt_lengths-1)*audio_len+audio_lengths-1
                            # attn_st_hypo = torch.scatter(attn_st_hypo.view(bsz,-1).contiguous(),-1,eos_idx.contiguous(),2).view(bsz, tgt_len, audio_len).contiguous()
                            # mmax,_ = torch.max(attn_st_hypo,dim=-1)
                            # mmin,_ = torch.min(attn_st_hypo,dim=-1)
                            # cha = mmax - mmin
                            # cha_mask = cha == 0
                            # cha = cha.masked_fill(cha_mask, float("-inf"))
                            # attn_st_hypo =(attn_st_hypo-mmin.unsqueeze(-1))/ cha.unsqueeze(-1)
                            
                            values, _ = torch.max(attn_st_hypo,dim=-1)
                            values_mask = values == 0
                            values = values.masked_fill(values_mask, float("-inf"))
                            attn_st_hypo =(attn_st_hypo / values.unsqueeze(-1)) * 5
                            attn_st_hypo_float = utils.softmax(attn_st_hypo, dim=(-1))
                            attn_st_hypo = attn_st_hypo_float.type_as(attn_st_hypo)
                           
                            attn_st_hypo.type_as(attn_st)
                            

                            # print('*********************max*******************')
                            # print(i)
                            # print(torch.max(attn_st_hypo_float,dim=-1))
                            # print('*********************min*******************')
                            # print(i)
                            # print(torch.min(attn_st_hypo_float,dim=-1))
                            # if i == 5:
                            #     assert 1==2
                            
                            
                            
                            attn_st_hypo_detach = attn_st_hypo.detach()
                            #欧式距离loss    
                            attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(trg_mask.unsqueeze(-1), 0)
                            attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(audio_mask.unsqueeze(1), 0)
                            attn_st = attn_st.masked_fill(trg_mask.unsqueeze(-1), 0)
                            attn_st = attn_st.masked_fill(audio_mask.unsqueeze(1), 0)

                            attn_st_hypo_detach = attn_st_hypo_detach.masked_fill(empty_mask.unsqueeze(1), 0)

                            # eos_idx = (mt_lengths-1)*audio_len+audio_lengths-1
                            # attn_st_hypo_detach = torch.scatter(attn_st_hypo_detach.view(bsz,-1).contiguous(),-1,eos_idx.contiguous(),1).view(bsz, tgt_len, audio_len).contiguous()
                            
                            attn_st = attn_st.masked_fill(empty_mask.unsqueeze(1), 0)
                            loss_attn = (attn_st_hypo_detach - attn_st).norm(dim=2)

                            
                            
                            loss_attn_sum = loss_attn_sum + loss_attn.sum()
                    



                    # type_loss.data = loss_kl.data
                    label_smoothed_nll_loss = label_smoothed_nll_loss + loss_attn_sum
                    # print(loss)
                    
                    # print(print(torch.max(attn_st)))
                    # print(print(torch.max(attn_st)))
                    # print('********************')
                    # # print(loss_kl.data)
                    # print(nll_loss_attn.data)
                    # print(label_smoothed_nll_loss_attn.data)
                    # print(nll_loss.data)
                    # print(label_smoothed_nll_loss.data)
                    # print(nll_loss_asr.data)
                    # print(label_smoothed_nll_loss_asr.data)
                    # print(nll_loss_mt.data)
                    # print(label_smoothed_nll_loss_mt.data)
                
                
            else:
                label_smoothed_nll_loss_mt, nll_loss_mt = self.compute_loss(model, net_output, sample, reduce=reduce)

        if sample["dataset_type"] == "st":
            source_ntokens = sample["source_ntokens"]
            target_ntokens = sample["target_ntokens"]
            target_ntokens_st = target_ntokens
            target_ntokens_mt = 0
            sample_size = sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        else:
            source_ntokens = 0
            target_ntokens = sample["ntokens"]
            target_ntokens_mt = target_ntokens
            target_ntokens_st = 0
            sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]

        nsentences = sample["target"].size(0)
        if sample["dataset_type"] == "st":
            if model.num_updates > self.start_kl and model.training:
                multi_ce_loss = label_smoothed_nll_loss + label_smoothed_nll_loss_asr + label_smoothed_nll_loss_mt
            else:
                multi_ce_loss = label_smoothed_nll_loss + label_smoothed_nll_loss_asr + label_smoothed_nll_loss_mt
            loss = multi_ce_loss + self.contrastive_weight * contrastive_loss
        else:
            loss = label_smoothed_nll_loss_mt

        logging_output = {
            'loss':loss.data,  
            'loss_attn':loss_attn_sum.data, 
            'nll_loss':nll_loss.data, 
            'contrastive_loss':contrastive_loss.data, 
            'source_ntokens':source_ntokens, 
            'target_ntokens':target_ntokens, 
            'target_ntokens_mt':target_ntokens_mt, 
            'target_ntokens_st':target_ntokens_st, 
            'ntokens':target_ntokens, 
            'nsentences':nsentences, 
            'sample_size':sample_size, 
            'nll_loss_asr':nll_loss_asr.data, 
            'nll_loss_mt':nll_loss_mt.data, 
            'st_nsentences':nsentences if sample['dataset_type'] != 'mt' else 0, 
            'mt_nsentences':nsentences if sample['dataset_type'] == 'mt' else 0}
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output['n_correct'] = utils.item(n_correct.data)
            logging_output['total'] = utils.item(total.data)

        return loss, sample_size, logging_output

    def compute_loss_attn(self, model, sample, attn_st, attn_mt, reduce=True):
        tag = sample['tag']
        bsz, tgt_len, src_len_st = attn_st.size()
        _, _, src_len_mt = attn_st.size()
        res = tag[0].new(bsz, tgt_len, src_len_st).fill_(0)
        attn_mt = attn_mt.transpose(1,2)
        for i in range(bsz):
            idx = torch.tensor(0)
            for j in range(src_len_mt):
                if tag[i][j] == 0:
                    continue
                tmp = attn_mt[i][j].repeat(tag[i][j],1)
                # print(tmp)
                print(idx,idx+tag[i][j])
                res[i,idx:idx+tag[i][j],] = tmp
                idx=idx+tag[i][j]
                # print(idx)
    
    def compute_loss_attn_no_mfa(self, model, sample,lprobs, target, reduce=True):
        lprobs = lprobs.contiguous()
        target = target.contiguous()
        # if self.ignore_prefix_size > 0:
        #     lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
        #     target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(lprobs,
          target,
          (self.eps),
          ignore_index=0,
          reduce=reduce)
        return loss, nll_loss

    def compute_loss_asr(self, model, sample, reduce=True):
        net_output, _ = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['prev_output_src_tokens'],
                            is_text_input=False, need_all_layer_attn=(self.need_all_layer_attn and model.training))
        attn_asr = None
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample['source']
        if len(net_output) > 1 and net_output[1] is not None and model.training:
            attn_asr = net_output[1]['attn'][0].detach()
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, 'batch_first', False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        # print(lprobs.size())
        # print(target.size())
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(lprobs,
          target,
          (self.eps),
          ignore_index=(self.padding_idx),
          reduce=reduce)
        return loss, nll_loss, attn_asr, net_output

    def compute_loss_mt(self, model, sample, reduce=True):
        net_output, _ = model((sample['source']), (sample['source_lengths']), (sample['net_input']['prev_output_tokens']),
          is_text_input=True, need_all_layer_attn=(self.need_all_layer_attn and model.training))
        attn_mt = None
        if len(net_output) > 1 and net_output[1] is not None and model.training:
            attn_mt = net_output[1]['attn'][0].detach()
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, 'batch_first', False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(lprobs,
          target,
          (self.eps),
          ignore_index=(self.padding_idx),
          reduce=reduce)
        return loss, nll_loss, attn_mt, net_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum((log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = sum((log.get('nll_loss', 0) for log in logging_outputs))
        contrastive_loss_sum = sum((log.get('contrastive_loss', 0) for log in logging_outputs))
        target_ntokens = sum((log.get('target_ntokens', 0) for log in logging_outputs))
        source_ntokens = sum((log.get('source_ntokens', 0) for log in logging_outputs))
        target_ntokens_mt = sum((log.get('target_ntokens_mt', 0) for log in logging_outputs))
        target_ntokens_st = sum((log.get('target_ntokens_st', 0) for log in logging_outputs))
        ntokens = sum((log.get('ntokens', 0) for log in logging_outputs))
        nsentences = sum((log.get('nsentences', 0) for log in logging_outputs))
        mt_nsentences = sum((log.get('mt_nsentences', 0) for log in logging_outputs))
        st_nsentences = sum((log.get('st_nsentences', 0) for log in logging_outputs))
        sample_size = sum((log.get('sample_size', 0) for log in logging_outputs))
        nll_loss_sum_asr = sum((log.get('nll_loss_asr', 0) for log in logging_outputs))
        nll_loss_sum_mt = sum((log.get('nll_loss_mt', 0) for log in logging_outputs))
        loss_attn = sum((log.get('loss_attn', 0) for log in logging_outputs))
        # print("**********reduce_metrics********************")
        # print(loss_attn)
        # print(nll_loss_sum)
        # print(nll_loss_sum_asr)
        # print(nll_loss_sum_mt)
        metrics.log_scalar('loss', (loss_sum / sample_size / math.log(2)),
          sample_size, round=3)
        metrics.log_scalar('nll_loss', (nll_loss_sum / target_ntokens_st / math.log(2)),
          target_ntokens_st, round=3)
        metrics.log_scalar('loss_attn', (loss_attn / target_ntokens_st / math.log(2)),
          target_ntokens_st, round=3)
        metrics.log_scalar('contrasitve_loss', (contrastive_loss_sum / st_nsentences / math.log(2)),
          st_nsentences, round=3)
        metrics.log_scalar('nll_loss_asr', (nll_loss_sum_asr / source_ntokens / math.log(2)),
          source_ntokens, round=3)
        metrics.log_scalar('nll_loss_mt', (nll_loss_sum_mt / target_ntokens / math.log(2)),
          target_ntokens, round=3)
        metrics.log_scalar('bsz_st', st_nsentences, priority=190, round=1)
        metrics.log_scalar('bsz_mt', mt_nsentences, priority=190, round=1)
        total = utils.item(sum((log.get('total', 0) for log in logging_outputs)))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
import torch
from fairseq.models.transformer import TransformerModel
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq import distributed_utils, utils
from fairseq.utils import print_r0
from hmoe.hmoe_encoder import HmoeTransformerEncoder
from hmoe.hmoe_decoder import HmoeTransformerDecoder
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import torch.nn as nn
import os
import logging
logger = logging.getLogger(__name__)

id2subset = [0, 2, 0, 2, 2, 1, 2, 2, 2, 0, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 0, 1, 0, 1, 1, 0, 2, 2, 0, 2, 1, 0, 2, 0, 1, 2, 1, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1, 0, 2, 2, 2, 2, 1, 0, 2, 2, 1, 2, 0, 1, 0, 2, 1, 2, 1, 2, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2, 2, 2]

def lookup_lang_group_labels(tgt_lang_id):
    dictionary = torch.tensor(id2subset).to(tgt_lang_id.device)
    labels = torch.index_select(dictionary, 0, tgt_lang_id.view(-1)).to(tgt_lang_id.device)
    
    return labels.view(-1,1)

class new_Mlp(nn.Module):
    def __init__(self, num_tasks, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.emb = nn.Embedding(num_tasks, in_features)
        with torch.no_grad():
            self.emb.weight.normal_(mean=0, std=1.0)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.emb(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x

@register_model("hmoe")
class HmoeTransformer(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.num_tasks = 101
        self.gate_task_specific_dim = 512

        self.global_lang_list = torch.arange(self.num_tasks, dtype=torch.int)

        if args.top_k_cnt:
            for subset, k in args.top_k_cnt.items():
                logger.warning(f'init hmoe {subset} top {k}')
        if args.use_task_emb:
            self.gate_task_represent = new_Mlp(
                num_tasks = self.num_tasks,
                in_features=self.gate_task_specific_dim, 
                hidden_features=self.gate_task_specific_dim, 
                out_features=self.gate_task_specific_dim,)
        else:
            self.gate_task_represent = None
            
    @classmethod
    def build_model(cls, args, task):
        args.top_k_cnt=task.top_k_cnt
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return HmoeTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return HmoeTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_lang_id,
        tgt_lang_id,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if (tgt_lang_id is not None) and (self.gate_task_represent is not None):

            subset_id = lookup_lang_group_labels(tgt_lang_id)
            tgt_lang_emb = self.gate_task_represent(tgt_lang_id)

            if self.args.add_lang_loss and self.training:
                global_lang_list = self.global_lang_list.to(tgt_lang_id.device)
                global_lang_emb = self.gate_task_represent(global_lang_list)
                global_lang_emb = global_lang_emb.view(global_lang_emb.shape[0],1,global_lang_emb.shape[1])
                tgt_lang_emb = torch.cat((tgt_lang_emb,global_lang_emb),dim=0)
                
            # tgt_lang_emb = tgt_lang_emb.squeeze(1)
        elif tgt_lang_id is not None:
            tgt_lang_emb = tgt_lang_id
        else:
            tgt_lang_emb = None
            
        if self.args.enable_encoder_token_routing:
            encoder_out = self.encoder(
                src_tokens=src_tokens, 
                src_lengths=src_lengths, 
                tgt_lang_id=None, 
                subset_id=subset_id,
                return_all_hiddens=return_all_hiddens
            )
        else:
            encoder_out = self.encoder(
                src_tokens=src_tokens, 
                src_lengths=src_lengths, 
                tgt_lang_id=tgt_lang_emb, 
                subset_id=subset_id,
                return_all_hiddens=return_all_hiddens
            )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            tgt_lang_id=tgt_lang_emb,
            subset_id=subset_id, 
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        
                    
        return decoder_out 
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--topk-threshold', type=float, default=0.,)
        parser.add_argument('--dynamic-expert-allocate', action='store_true', default=False, help="")
        parser.add_argument('--add-lang-loss', action='store_true', default=False, help="")
        parser.add_argument('--enable-tgt-routing', action='store_true', default=False, help="")
        parser.add_argument('--enable-sent-routing', action='store_true', default=False, help="")
        parser.add_argument('--enable-encoder-token-routing', action='store_true', default=False, help="")
        parser.add_argument('--use-task-emb', action='store_true', default=False, help="")
        parser.add_argument('--hmoe-gate', action='store_true', default=False, help="")
        parser.add_argument('--hrmoe-gate', action='store_true', default=False, help="")
        parser.add_argument('--temperature', type=float, default=1.0, help="temperature for df-gate, only make sense if df-gate-type=sigmoid")
        parser.add_argument('--reorder-layers', action='store_true', default=False, help="")
        parser.add_argument('--gate-type', type=str, default='softmax',
            choices=['softmax','sigmoid',])
        parser.add_argument('--layernorm-after-moe-layer', action='store_true', default=False)
        TransformerModel.add_args(parser)        

@register_model_architecture('hmoe', 'hmoe')
def hmoe(args):
    pass

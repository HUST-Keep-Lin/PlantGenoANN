import os
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .modeling_caduceus_moe import CaduceusPh, CaduceusPhPreTrainedModel
from .configuration_caduceus_ph import CaduceusPhConfig

class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dilation, groups=1):
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=padding, 
                dilation=dilation, 
                groups=groups,
            ),
            nn.SiLU(),
            nn.Dropout1d(p=0.25),
        )

    def forward(self, x: torch.Tensor):

        return self.dilated_conv(x)
    
class NormLayer(nn.Module):
    def __init__(self, norm_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=norm_shape)

    def forward(self, x: torch.Tensor):
        x = self.layer_norm(x.transpose(1,2))

        return x.transpose(1,2)
    
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dilation):
        super().__init__()
        self.dilated_layer = DilatedConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        self.norm = NormLayer(out_channels)

    def forward(self, x: torch.Tensor):
        y = self.dilated_layer(x)
        y = self.norm(y)

        return y
    
class ShiftedConvBlock(nn.Module):
    def __init__(self, n_layer, in_channels, out_channels, padding, dilation, shift_steps=2, shift_stride=1):
        super().__init__()
        self.shift_steps = shift_steps
        self.hf_shift_steps = shift_steps // 2
        self.shift_stride = shift_stride
        self.pad = (shift_steps*shift_stride) // 2

        first_layer = [
            DilatedConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                dilation=dilation,
                groups=shift_steps
            )
        ]
        next_layer = [
            DilatedConvLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                padding=padding,
                dilation=dilation,
                groups=shift_steps,
            ) 
            for i in range(n_layer-1)
        ]

        self.shifted_layers = nn.ModuleList(first_layer+next_layer)
        self.norm = NormLayer(out_channels)

    def shift(self, x: torch.Tensor):
        bsz, d, L = x.shape
        xn = F.pad(x, (self.pad ,self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_steps, 1)
        x_shift = [torch.roll(x_c, shift*self.shift_stride, 2) for x_c, shift in zip(xs, range(-self.hf_shift_steps, self.hf_shift_steps+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, L)

        return x_cat

    def forward(self, x: torch.Tensor):
        y = self.shift(x)
        for layer in self.shifted_layers:
            y = layer(y)
        y = self.norm(y)

        return y
    
class ShiftedUNetHead(nn.Module):
    def __init__(
            self,
            embd_dim=[512,1024,1536,2560,4096],
            dilation=[1,4,8,16],
            shift_steps=[2,4,4]
        ):
        super().__init__()
        self.down_conv1 = DilatedConvBlock(embd_dim[0], embd_dim[1], padding=dilation[0], dilation=dilation[0])

        self.down_conv2 = ShiftedConvBlock(
            n_layer=2, in_channels=embd_dim[1], out_channels=embd_dim[2], 
            padding=dilation[1], dilation=dilation[1], shift_steps=shift_steps[0]
        )
        self.down_conv3 = ShiftedConvBlock(
            n_layer=2, in_channels=embd_dim[2], out_channels=embd_dim[3], 
            padding=dilation[2], dilation=dilation[2], shift_steps=shift_steps[1]
        )
        self.down_conv4 = ShiftedConvBlock(
            n_layer=3, in_channels=embd_dim[3], out_channels=embd_dim[4], 
            padding=dilation[3], dilation=dilation[3], shift_steps=shift_steps[2]
        )

        self.up_trans1 = nn.ConvTranspose1d(embd_dim[4], embd_dim[3], kernel_size=2, stride=2, groups=128)
        self.up_trans2 = nn.ConvTranspose1d(embd_dim[3], embd_dim[2], kernel_size=2, stride=2, groups=128)
        self.up_trans3 = nn.ConvTranspose1d(embd_dim[2], embd_dim[1], kernel_size=2, stride=2, groups=128)
        
        self.up_conv1 = ShiftedConvBlock(
            n_layer=2, in_channels=embd_dim[3], out_channels=embd_dim[3],
            padding=dilation[2], dilation=dilation[2], shift_steps=shift_steps[1]
        )
        self.up_conv2 = ShiftedConvBlock(
            n_layer=2, in_channels=embd_dim[2], out_channels=embd_dim[2],
            padding=dilation[1], dilation=dilation[1], shift_steps=shift_steps[0]
        )
        self.up_conv3 = DilatedConvBlock(embd_dim[1], embd_dim[1], padding=dilation[0], dilation=dilation[0])

        self.norm_f = NormLayer(embd_dim[1])

    def forward(self, x: torch.Tensor): 
        x = self.down_conv1(x) 
        t1 = x

        x = F.avg_pool1d(x, kernel_size=2, stride=2) 
        x = self.down_conv2(x) 
        t3 = x

        x = F.avg_pool1d(x, kernel_size=2, stride=2) 
        x = self.down_conv3(x) 
        t5 = x

        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = self.down_conv4(x)

        x = self.up_trans1(x) 
        x = torch.add(x, t5) 
        x = self.up_conv1(x) 

        x = self.up_trans2(x) 
        x = torch.add(x, t3) 
        x = self.up_conv2(x) 

        x = self.up_trans3(x) 
        x = torch.add(x, t1) 
        x = self.up_conv3(x) 
        
        return self.norm_f(x)

class SegmentCaduceus(CaduceusPhPreTrainedModel):
    """SegmentCaduceusModel for sequence segmentation"""

    def __init__(self, config: CaduceusPhConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_features = 10
        self.training_features = None
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus_ph = CaduceusPh(config, **factory_kwargs, **kwargs)
        
        self.shift_unet_head = ShiftedUNetHead()
        self.final_head = nn.Sequential(
            nn.Conv1d(2 * config.d_model, config.d_model, kernel_size=1, padding=0),
            nn.SiLU(),
            nn.Conv1d(config.d_model, self.num_features, kernel_size=1, padding=0),
        )
        self.post_init()

    def reset_loss_weight(self, loss_weight):
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weight))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # **kwargs
    ):
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.caduceus_ph(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.shift_unet_head(outputs[0][:,1:-1,:].transpose(1,2))
        logits = self.final_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
        )

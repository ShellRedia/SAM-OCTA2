import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from dsconv import MultiView_DSConv
import torch.utils.checkpoint as checkpoint

# LoCon
class LoConLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.conv_a = nn.Conv2d(in_dim, rank, kernel_size=3, padding=1, bias=False)
        self.conv_b = nn.Conv2d(rank, out_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.conv_b(self.conv_a(x))
        out = out.permute(0, 2, 3, 1)
        return out

# LoDS
class LoDSLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.dsconv_a = MultiView_DSConv(in_dim, rank)
        self.dsconv_b = MultiView_DSConv(rank, out_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.dsconv_b(self.dsconv_a(x))
        out = out.permute(0, 2, 3, 1)
        return out
    
# LoRA
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight) 

    def forward(self, x):
        return self.lora_b(self.lora_a(x))

# LoKR
class LoKRLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=-1): # rank 在基础 LoKR 中可选
        super().__init__()
        
        self.o1, self.o2 = self._get_factors(out_dim)
        self.i1, self.i2 = self._get_factors(in_dim)
        
        self.w1 = nn.Parameter(torch.empty(self.o1, self.i1))
        self.w2 = nn.Parameter(torch.empty(self.o2, self.i2))

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.zeros_(self.w2)

    def _get_factors(self, dim):
        m = int(math.sqrt(dim))
        while dim % m != 0:
            m -= 1
        return m, dim // m

    def forward(self, x):
        weight = torch.kron(self.w1, self.w2)

        return F.linear(x, weight)


class AdapterAttach(nn.Module):
    def __init__(self, original_block, rank, in_dim, out_dim, adapter_type):
        super().__init__()
        self.block = original_block
        self.adapter = nn.Identity()

        if adapter_type=="LoCon":
            self.adapter = LoConLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank
            )
        elif adapter_type=="LoDS":
            self.adapter = LoDSLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank
            )
        elif adapter_type=="LoRA":
            self.adapter = LoRALayer(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank
            )
        elif adapter_type=="LoKR":
            self.adapter = LoKRLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=rank
            )
            

    def forward(self, x):
        out_block = self.block(x)

        # original resolution
        out_adapter = self.adapter(x)
        
        if out_adapter.shape[1:3] != out_block.shape[1:3]:
            out_adapter = out_adapter.permute(0, 3, 1, 2)
            out_adapter = F.interpolate(
                out_adapter, 
                size=(out_block.shape[1], out_block.shape[2]), 
                mode='nearest'
            )
            out_adapter = out_adapter.permute(0, 2, 3, 1)
            
        return out_block + out_adapter
        

class SAM2_Adapter:
    def __init__(self):
        pass

    def attach_adapter(self, sam2_model, rank, adapter_type):
        all_blocks = sam2_model.image_encoder.trunk.blocks
        total_len = len(all_blocks)


        device = next(sam2_model.parameters()).device

        adapted_layer_idxs = {0,1,2,3,5,6,21,22} if total_len == 24 else {0,1,2,3,8,9,44,45}

        current_in_dim = all_blocks[0].dim
        dummy_res = 32
        dummy_x = torch.zeros(1, dummy_res, dummy_res, current_in_dim).to(device)
        
        for i in range(total_len):
            old_blk = all_blocks[i]

            with torch.no_grad():
                dummy_out = old_blk(dummy_x)
                current_out_dim = dummy_out.shape[-1]

            if i in adapted_layer_idxs:
                all_blocks[i] = AdapterAttach(
                    old_blk, rank, current_in_dim, current_out_dim, adapter_type
                )
                
            dummy_x = dummy_out
            current_in_dim = current_out_dim

        for blk in sam2_model.image_encoder.trunk.blocks:
            self.checkpoint_wrapper(blk)
                
        return sam2_model

    def checkpoint_wrapper(self, module):
        original_forward = module.forward
        def new_forward(*args, **kwargs):
            return checkpoint.checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
        module.forward = new_forward
        
    def check_image_encoder_structure(self, sam2_model):
        summary(sam2_model.image_encoder, (1, 3, 1024, 1024), device="cuda")


# if __name__ == "__main__":
#     sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
#     lora_sam = LoRA_Sam(sam, 4)
#     lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
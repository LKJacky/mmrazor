import torch
import torch.nn as nn
from mmengine import Config
from mmrazor.registry import MODELS
from dms_deit import DeitDms

config_path='configs/prune_deit.py'
path='work_dirs/prune_deit_lr/epoch_30.pth'

config=Config.fromfile(config_path)
model:DeitDms=MODELS.build(config['model'])

state=torch.load(path,map_location='cpu')
state_dict=state['state_dict']

model.load_state_dict(state_dict)

for unit in model.mutator.dtp_mutator.mutable_units:
    unit.mutable_channel.e.data.fill_(0.5)
    unit.mutable_channel.sync_mask()

for block  in model.mutator.block_mutables:
    block.e.data.fill_(11/16)
    block.sync_mask()
    
for attn_mutables in model.mutator.attn_mutables:
    head,qk,v=attn_mutables.values()
    head.e.data.fill_(0.5)
    head.sync_mask()
    qk.e.data.fill_(1.0)
    qk.sync_mask()
    v.e.data.fill_(1.0)
    v.sync_mask()
state['state_dict']=model.state_dict()
torch.save(state,'tmp.pth')
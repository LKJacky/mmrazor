_base_ = 'mmcls::deit/deit-tiny_pt-4xb256_in1k.py'

train_dataloader = dict(batch_size=32)
test_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=32)

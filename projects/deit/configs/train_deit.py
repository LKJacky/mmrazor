_base_ = 'mmcls::deit/deit-small_pt-4xb256_in1k.py'

model = dict(backbone=dict(drop_path_rate=0.0, type='VisionTransformer2'), )

# train_dataloader = dict(batch_size=32)
# test_dataloader = dict(batch_size=32)
# val_dataloader = dict(batch_size=32)

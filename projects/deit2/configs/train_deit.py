_base_ = 'mmcls::deit/deit-small_pt-4xb256_in1k.py'

model = dict(backbone=dict(drop_path_rate=0.0, type='VisionTransformer2'), )

# train_dataloader = dict(batch_size=32)
# test_dataloader = dict(batch_size=32)
# val_dataloader = dict(batch_size=32)



# default_hooks = dict(
#     checkpoint=dict(
#         type='CheckpointHook',
#         interval=1,
#         save_best='auto',
#         max_keep_ckpts=5,
#     ), )


# train_dataloader = _base_.train_dataloader
# val_dataloader = _base_.val_dataloader
# test_dataloader = _base_.test_dataloader

# train_dataloader.batch_size = 64
# val_dataloader.batch_size = 64

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/coco':
#         's3://openmmlab/datasets/detection/coco',
#         'data/coco':
#         's3://openmmlab/datasets/detection/coco',
#         './data/cityscapes':
#         's3://openmmlab/datasets/segmentation/cityscapes',
#         'data/cityscapes':
#         's3://openmmlab/datasets/segmentation/cityscapes',
#         './data/imagenet':
#         's3://openmmlab/datasets/classification/imagenet',
#         'data/imagenet':
#         's3://openmmlab/datasets/classification/imagenet'
#     }))
# train_dataloader['dataset']['pipeline'][0][
#     'file_client_args'] = file_client_args
# val_dataloader['dataset']['pipeline'][0]['file_client_args'] = file_client_args
# test_dataloader['dataset']['pipeline'][0][
#     'file_client_args'] = file_client_args

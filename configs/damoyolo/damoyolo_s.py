_base_ = ['mmyolo::damoyolo/damo_yolo_s.py', './supernet.py']

model = dict(
    backbone=dict(
        _delete_=True,
        _scope_='mmrazor',
        type='SearchAableModelDeployWrapper',
        architecture=dict(
            type='TinyNasBackbone',
            structure_info=_base_.SUPERNET,
        ),
        to_static=True,
        subnet_dict='./configs/damoyolo/L25.yaml',
    ))

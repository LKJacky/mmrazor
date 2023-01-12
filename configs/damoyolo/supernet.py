SUPERNET = [
    dict(
        type='SearchableFocus',
        in_channels=3,
        out_channels=1024,
        ksize=3,
        stride=1,
        act='relu',
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            stride=1,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
        with_spp=True,
    )
]

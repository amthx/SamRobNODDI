# model settings

model_cfg = dict(
    backbone=dict(type='ResNet',
                  in_channel=56,
                  hidden=64,
                  out_channel=3)
)


# train
data_cfg = dict(
    batch_size=4,
    num_workers=32,
    is_voxel=False,
    train=dict(
        pretrained_flag=False,
        pretrained_weights='./parameter_ResNet_repeat/40subject_Nb0_sh_w1_1_w2_1_w3_0001/lr0.001/epoch_**.pth',
        epoches=100,
    ),
    test=dict(
        batch_size=4
    )
)

# optimizer
optimizer_cfg = dict(
    type='Adam',
    lr=0.001)

# learning
lr_config = dict(is_scheduler=False,
                 type='StepLR',
                 step=5,
                 gamma=0.5)


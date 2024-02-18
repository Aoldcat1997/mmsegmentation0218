# optimizer
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.1)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=60000,
        begin=1000,
        end=80000,
        by_epoch=False)
]
# training schedule for GLC
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,save_best = "mIoU",save_last = True,
                    max_keep_ckpts = 5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',draw = True,interval = 1))
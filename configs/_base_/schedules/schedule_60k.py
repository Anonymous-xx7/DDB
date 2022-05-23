_base_ = ['schedule_40k.py']
runner = dict(max_iters=60000)
# Logging Configuration
checkpoint_config = dict(interval=6000)
evaluation = dict(interval=6000)













'''###____pretty_text____###'''



'''
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=6000, max_keep_ckpts=1)
evaluation = dict(interval=6000, metric='mIoU', save_best='mIoU')
'''

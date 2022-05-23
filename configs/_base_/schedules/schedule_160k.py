_base_ = ['schedule_40k.py']
runner = dict(max_iters=160000)
# Logging Configuration
checkpoint_config = dict(interval=16000)
evaluation = dict(interval=8000)













'''###____pretty_text____###'''



'''
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
'''

# Baseline UDA
uda = dict(
    type='DKD',
    pseudo_threshold=0.968,
    mix_loss=False,
    teacher_model_cfg=None,
    cu_model_load_from='',
    ca_model_load_from='',
    stu_model_load_from=None,
    soft_distill=False,
    soft_distill_w=0.5,
    proto_rectify=False,
    rectify_on_prob=True,
    proto_momentum=0.9999,
    use_pl_weight=False,
    temp=1,
    cu_proto_path='',
    ca_proto_path='',
    debug_img_interval=1000)













'''###____pretty_text____###'''



'''
uda = dict(
    type='DKD',
    pseudo_threshold=0.968,
    mix_loss=False,
    teacher_model_cfg=None,
    cu_model_load_from='',
    ca_model_load_from='',
    stu_model_load_from=None,
    soft_distill=False,
    soft_distill_w=0.5,
    proto_rectify=False,
    rectify_on_prob=True,
    proto_momentum=0.9999,
    use_pl_weight=False,
    temp=1,
    cu_proto_path='',
    ca_proto_path='',
    debug_img_interval=1000)
'''

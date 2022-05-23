_base_ = [
    "../../_base_/datasets/cityscapes_half_512x512.py",
    "../../_base_/models/deeplabv2red_r101v1c-d8_adapter.py",
]
module_cfg = {{_base_.model}}
model = dict(
    _delete_=True,
    type="Tester",
    segmentor=module_cfg,
    weight="weights/gta2cs/source_only/weight.pth",
)

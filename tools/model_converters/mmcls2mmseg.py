from collections import OrderedDict
import torch

if __name__ == '__main__':
    ckpt_dir = 'checkpoints/resnet/densecl_resnet50_8xb32-coslr-200e_in1k_20220225-8c7808fe.pth'
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'backbone' in k:
            new_state_dict[k.replace('backbone.', '')] = v
        else:
            new_state_dict[k] = v
    torch.save(new_state_dict,'checkpoints/resnet/resnet50.pth')
    pass

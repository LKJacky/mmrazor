import argparse
from timm import create_model
import torch
import ptflops
import torch.nn as nn
from dms_eff import EffDmsAlgorithm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='efficientnet_b7')
parser.add_argument('--alg', type=str, default='')
parser.add_argument('--sub_alg', type=str, default='')
parser.add_argument('--target', type=float, default=0.8)
parser.add_argument('--scale', type=str, default='false')
parser.add_argument('--size', type=int, default=224)
args = parser.parse_args()

defautl_alg_args = dict(
    mutator_kwargs=dict(
        dtp_mutator_cfg=dict(
            parse_cfg=dict(
                demo_input=dict(input_shape=(1, 3, args.size,
                                             args.size), ), )), ),
    scheduler_kargs=dict(flops_target=args.target))


def load_algo(model: nn.Module, algo_path: str):
    model: nn.Module = EffDmsAlgorithm(model, **defautl_alg_args)
    if algo_path != "":
        state = torch.load(algo_path, map_location='cpu')['state_dict']
        model.load_state_dict(state)
    model = model.to_static_model(scale=args.scale == 'true')
    return model


if __name__ == "__main__":

    model_name = args.model

    model = create_model(model_name)
    model = load_algo(model, args.alg)
    if args.sub_alg != '':
        model = load_algo(model, args.sub_alg)

    res = ptflops.get_model_complexity_info(
        model, (3, args.size, args.size),
        print_per_layer_stat=False,
        custom_modules_hooks={})

    model = EffDmsAlgorithm(model, **defautl_alg_args)
    print(model.mutator.info())
    print(res)

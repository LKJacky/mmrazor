import argparse
from timm import create_model
import torch
import ptflops
import torch.nn as nn
from dms_eff import EffDmsAlgorithm


def load_algo(model: nn.Module, algo_path: str):
    state = torch.load(algo_path, map_location='cpu')['state_dict']
    model = EffDmsAlgorithm(model)
    model.load_state_dict(state)
    model = model.to_static_model()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--alg', type=str, default='')
    parser.add_argument('--sub_alg', type=str, default='')
    args = parser.parse_args()

    model_name = args.model

    model = create_model(model_name)
    if args.alg != '':
        model = load_algo(model, args.alg)
        if args.sub_alg != '':
            model = load_algo(model, args.sub_alg)

    res = ptflops.get_model_complexity_info(
        model, (3, 224, 224),
        print_per_layer_stat=False,
        custom_modules_hooks={})

    model = EffDmsAlgorithm(model)
    print(model.mutator.info())
    print(res)

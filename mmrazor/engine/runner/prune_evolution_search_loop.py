# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine import fileio
from mmengine.dist import broadcast_object_list
from mmengine.evaluator import Evaluator
from torch.utils.data import DataLoader

from mmrazor.models.task_modules import ResourceEstimator
from mmrazor.registry import LOOPS
from mmrazor.structures import Candidates
from mmrazor.utils import SupportRandomSubnet
from .evolution_search_loop import EvolutionSearchLoop


def get_flops(model: nn.Module, subnet: SupportRandomSubnet,
              estimator: ResourceEstimator):
    """Check whether is beyond flops constraints.

    Returns:
        bool: The result of checking.
    """

    assert hasattr(model, 'set_subnet') and hasattr(model, 'architecture')
    model = copy.deepcopy(model)
    model.set_subnet(subnet)

    model_to_check = model.architecture

    results = estimator.estimate(model=model_to_check)

    flops = results['flops']
    return flops


def auto_scale(subnet, target, now):
    new_subnet = copy.deepcopy(subnet)
    scale = math.sqrt(target / now)
    for key in new_subnet:
        new_subnet[key] = max(min(new_subnet[key] * scale, 1.0), 0.01)
    return new_subnet


@LOOPS.register_module()
class PruneEvolutionSearchLoop(EvolutionSearchLoop):
    """Loop for evolution searching.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        max_epochs (int): Total searching epochs. Defaults to 20.
        max_keep_ckpts (int): The maximum checkpoints of searcher to keep.
            Defaults to 3.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching.
        num_candidates (int): The length of candidate pool. Defaults to 50.
        top_k (int): Specify top k candidates based on scores. Defaults to 10.
        num_mutation (int): The number of candidates got by mutation.
            Defaults to 25.
        num_crossover (int): The number of candidates got by crossover.
            Defaults to 25.
        mutate_prob (float): The probability of mutation. Defaults to 0.1.
        flops_range (tuple, optional): It is used for screening candidates.
        resource_estimator_cfg (dict): The config for building estimator, which
            is be used to estimate the flops of sampled subnet. Defaults to
            None, which means default config is used.
        score_key (str): Specify one metric in evaluation results to score
            candidates. Defaults to 'accuracy_top-1'.
        init_candidates (str, optional): The candidates file path, which is
            used to init `self.candidates`. Its format is usually in .yaml
            format. Defaults to None.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 bn_dataloader,
                 evaluator: Union[Evaluator, Dict, List],
                 max_epochs: int = 20,
                 max_keep_ckpts: int = 3,
                 resume_from: Optional[str] = None,
                 num_candidates: int = 50,
                 top_k: int = 10,
                 num_mutation: int = 25,
                 num_crossover: int = 25,
                 mutate_prob: float = 0.1,
                 flops_range: Tuple[float, float] = (0.1, 0.9),
                 resource_estimator_cfg: Optional[dict] = None,
                 score_key: str = 'accuracy/top1',
                 init_candidates: Optional[str] = None) -> None:
        if bn_dataloader['batch_size'] < 2:
            bn_dataloader['batch_size'] = 2

        super().__init__(runner, dataloader, evaluator, max_epochs,
                         max_keep_ckpts, resume_from, num_candidates, top_k,
                         num_mutation, num_crossover, mutate_prob, flops_range,
                         resource_estimator_cfg, score_key, init_candidates)
        if isinstance(bn_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.bn_dataloader = runner.build_dataloader(
                bn_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.bn_dataloader = bn_dataloader

        if self.runner.distributed:
            self.search_wrapper = runner.model.module
        else:
            self.search_wrapper = runner.model

        # bn rescale

    def run(self) -> None:
        max_iter = 100
        iter = 0
        self.search_wrapper.eval()
        for _, data_batch in enumerate(self.bn_dataloader):
            data = self.search_wrapper.data_preprocessor(data_batch, True)
            self.search_wrapper._run_forward(
                data, mode='tensor')  # type: ignore
            iter += 1
            if iter > max_iter:
                break

        self.flops_range: Tuple[float, float] = self._update_flop_range()
        self.min_flops = self._min_flops()
        assert self.min_flops < self.flops_range[0], 'Cannot reach flop targe.'

        return super().run()

    def _min_flops(self):
        subnet = self.model.sample_subnet()
        for key in subnet:
            subnet[key] = 0.001
        flops = get_flops(self.model, subnet, self.estimator)
        return flops

    def run_epoch(self) -> None:
        super().run_epoch()
        self._save_best_fix_subnet()

    def sample_candidates(self) -> None:
        """Update candidate pool contains specified number of candicates."""
        if self.runner.rank == 0:
            while len(self.candidates) < self.num_candidates:
                candidate = self.model.sample_subnet()
                passed, candidate = self._scale_and_check_subnet_constraints(
                    random_subnet=candidate)
                if passed:
                    self.candidates.append(candidate)
        else:
            self.candidates = Candidates([None] * self.num_candidates)
        # broadcast candidates to val with multi-GPUs.
        broadcast_object_list(self.candidates.data)

    def gen_mutation_candidates(self) -> List:
        """Generate specified number of mutation candicates."""
        mutation_candidates: List = []
        max_mutate_iters = self.num_mutation * 10
        mutate_iter = 0
        while len(mutation_candidates) < self.num_mutation:
            mutate_iter += 1
            if mutate_iter > max_mutate_iters:
                break

            mutation_candidate = self._mutation()

            passed, candidate = self._scale_and_check_subnet_constraints(
                random_subnet=mutation_candidate)
            if passed:
                mutation_candidates.append(candidate)
        return mutation_candidates

    def gen_crossover_candidates(self) -> List:
        """Generate specofied number of crossover candicates."""
        crossover_candidates: List = []
        crossover_iter = 0
        max_crossover_iters = self.num_crossover * 10
        while len(crossover_candidates) < self.num_crossover:
            crossover_iter += 1
            if crossover_iter > max_crossover_iters:
                break

            crossover_candidate = self._crossover()

            passed, candidate = self._scale_and_check_subnet_constraints(
                random_subnet=crossover_candidate)
            if passed:
                crossover_candidates.append(candidate)
        return crossover_candidates

    def _save_best_fix_subnet(self):
        """Save best subnet in searched top-k candidates."""
        if self.runner.rank == 0:
            best_random_subnet = self.top_k_candidates.subnets[0]
            self.model.set_subnet(best_random_subnet)
            save_name = 'best_fix_subnet.json'
            fileio.dump(
                best_random_subnet,
                osp.join(self.runner.work_dir, save_name),
                indent=4)
            self.runner.logger.info(
                'Search finished and '
                f'{save_name} saved in {self.runner.work_dir}.')

    @torch.no_grad()
    def _val_candidate(self) -> Dict:
        # bn rescale
        max_iter = 100
        iter = 0
        self.search_wrapper.train()
        for _, data_batch in enumerate(self.bn_dataloader):
            data = self.search_wrapper.data_preprocessor(data_batch, True)
            self.search_wrapper._run_forward(
                data, mode='tensor')  # type: ignore
            iter += 1
            if iter > max_iter:
                break
        if self.score_key == 'loss':
            return self._val_by_loss()
        else:
            return super()._val_candidate()

    def _scale_and_check_subnet_constraints(
            self,
            random_subnet: SupportRandomSubnet,
            auto_scale_times=20) -> Tuple[bool, SupportRandomSubnet]:
        """Check whether is beyond constraints.

        Returns:
            bool: The result of checking.
        """
        is_pass = False
        assert auto_scale_times >= 0
        for _ in range(auto_scale_times + 1):
            flops = get_flops(self.model, random_subnet, self.estimator)
            if self.check_subnet_flops(flops):
                is_pass = True
                break
            else:
                random_subnet = auto_scale(
                    random_subnet,
                    (self.flops_range[1] + self.flops_range[0]) / 2, flops)
                continue
        if is_pass:
            from mmengine import MMLogger
            MMLogger.get_current_instance().info(
                f'sample a net,{flops},{self.flops_range}')
        else:
            from mmengine import MMLogger
            MMLogger.get_current_instance().info(
                f'sample a net failed,{flops},{self.flops_range}')
        return is_pass, random_subnet

    def _update_flop_range(self):
        flops = get_flops(self.model, self.model.curent_subnet(),
                          self.estimator)
        flops_range = [ratio * flops for ratio in self.flops_range]
        return flops_range

    def check_subnet_flops(self, flops):
        return self.flops_range[0] <= flops <= self.flops_range[
            1]  # type: ignore

    def _val_by_loss(self):
        from mmengine.dist import all_reduce
        self.runner.model.eval()
        sum_loss = 0
        for data_batch in self.dataloader:
            data = self.search_wrapper.data_preprocessor(data_batch)
            losses = self.search_wrapper.forward(**data, mode='loss')
            parsed_losses, _ = self.search_wrapper.parse_losses(
                losses)  # type: ignore
            sum_loss = sum_loss + parsed_losses
        all_reduce(sum_loss)
        return {'loss': sum_loss.item() * -1}

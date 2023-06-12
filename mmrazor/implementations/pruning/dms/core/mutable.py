# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.models.mutables import (BaseMutable, DerivedMutable,
                                     MutableChannelContainer,
                                     SimpleMutableChannel)

BlockThreshold = 0.5

MaskThreshold = 0.5
# dtp with taylor importance base dtp with adaptive importance


@torch.jit.script
def dtopk(x: torch.Tensor, e: torch.Tensor, lamda: float = 1.0):
    # add min or max
    # e = soft_clip(e, 1 / x.numel(), 1.0)

    y: torch.Tensor = -(x - e) * x.numel() * lamda
    s = y.sigmoid()
    return s


@torch.jit.script
def dtp_get_importance(v: torch.Tensor,
                       e: torch.Tensor,
                       lamda: float = 1.0,
                       space_min: float = 0,
                       space_max: float = 1.0):
    vm = v.unsqueeze(-1) - v.unsqueeze(-2)
    vm = (vm >= 0).float() - vm.detach() + vm
    v_union = vm.mean(dim=-1)  # big to small
    v_union = 1 - v_union
    if space_max != 1.0 or space_min != 0:
        v_union = v_union * (space_max - space_min) + space_min
    imp = dtopk(v_union, e, lamda=lamda)  # [0,1]
    return imp


def taylor_backward_hook_wrapper(module: 'DTPTMutableChannelImp', input):

    def taylor_backward_hook(grad):
        with torch.no_grad():
            module.update_taylor(input, grad)

    return taylor_backward_hook


#############################################################################


class DrivedDTPMutableChannelImp(DerivedMutable):

    def __init__(self,
                 choice_fn,
                 mask_fn,
                 expand_ratio,
                 source_mutables=None,
                 alias=None,
                 init_cfg=None) -> None:
        super().__init__(choice_fn, mask_fn, source_mutables, alias, init_cfg)
        self.expand_ratio = expand_ratio

    @property
    def current_imp(self):
        mutable = list(self.source_mutables)[0]
        mask = mutable.current_imp
        mask = torch.unsqueeze(
            mask,
            -1).expand(list(mask.shape) + [self.expand_ratio]).flatten(-2)
        return mask

    @property
    def current_imp_flop(self):
        mutable = list(self.source_mutables)[0]
        mask = mutable.current_imp_flop
        mask = torch.unsqueeze(
            mask,
            -1).expand(list(mask.shape) + [self.expand_ratio]).flatten(-2)
        return mask


class DMSMutableMixIn():

    def _dms_mutable_mixin_init(self, num_elem):

        self.use_tayler = True

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        taylor = torch.zeros([num_elem])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.lda = 1.0
        self.requires_grad_(False)

        self.grad_scaler = 1

    @property
    def current_imp(self):
        if self.taylor.max() == self.taylor.min():
            e_imp = torch.ones_like(self.taylor, requires_grad=True)
        else:
            e_imp = dtp_get_importance(self.taylor, self.e, lamda=self.lda)
        if self.training and e_imp.requires_grad and self.use_tayler:
            e_imp.register_hook(
                taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= MaskThreshold).float()
        return e_imp

    @property
    def current_imp_flop(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        return e_imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.taylor.numel() / 2, 1.0)

    @torch.no_grad()
    def update_taylor(self, input, grad):
        if self.grad_scaler != 0:
            grad = grad.float() / self.grad_scaler
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        if (not new_taylor.isnan().any()) and (not new_taylor.isinf().any()):
            if new_taylor.max() != new_taylor.min():
                self.taylor = self.taylor * self.decay + (
                    1 - self.decay) * new_taylor

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def activated_channels(self):
        return self.mask.bool().sum().item()

    def info(self):
        return (f'taylor: {self.taylor.min().item():.3f}\t'
                f'{self.taylor.max().item():.3f}\t'
                f'{self.taylor.min()==self.taylor.max()}\t'
                f'e: {self.e.item():.3f}')  # noqa

    def expand_mutable_channel(self, expand_ratio):

        def _expand_mask():
            mask = self.current_mask
            mask = torch.unsqueeze(
                mask, -1).expand(list(mask.shape) + [expand_ratio]).flatten(-2)
            return mask

        return DrivedDTPMutableChannelImp(_expand_mask, _expand_mask,
                                          expand_ratio, [self])

    @torch.no_grad()
    def to_index_importance(self):
        self.use_tayler = False
        self.taylor.data = 1 - torch.linspace(
            0, 1, self.taylor.numel(), device=self.taylor.device)


class DTPTMutableChannelImp(SimpleMutableChannel, DMSMutableMixIn):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self._dms_mutable_mixin_init(self.num_channels)

    def fix_chosen(self, chosen=None):
        return super().fix_chosen(chosen)


class ImpMutableChannelContainer(MutableChannelContainer):

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.register_buffer(
            '_tmp_imp', torch.ones([self.num_channels]), persistent=False)
        self._tmp_imp: torch.Tensor

    @property
    def current_imp(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return self._tmp_imp
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp for mutable in mutable_channels]
            if len(imps) == 1:
                return imps[0]
            else:
                imp = torch.cat(imps)
                return imp

    @property
    def current_imp_flop(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return self._tmp_imp
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp_flop for mutable in mutable_channels]
            if len(imps) == 1:
                imp = imps[0]
            else:
                imp = torch.cat(imps)
            return imp


#############################################################################


class MutableBlocks(BaseMutable, DMSMutableMixIn):

    def __init__(self, num_blocks) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        mask = torch.ones([num_blocks])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor

        self._dms_mutable_mixin_init(num_blocks)

        self.lda = 4.0
        self.flop_scale_converter = None

    def block_scale_fun_wrapper(self, i):

        def scale():
            scale = self.current_imp[i]
            if self.flop_scale_converter is None:
                return scale
            else:
                return self.flop_scale_converter(scale)

        return scale

    def block_flop_scale_fun_wrapper(self, i):

        def scale():
            return self.current_imp[i]

        return scale

    def info(self):

        def get_mask_str():
            mask_str = ''
            for i in range(self.num_blocks):
                if self.mask[i] == 1:
                    mask_str += '1'
                else:
                    mask_str += '0'
            return mask_str

        return (
            f'mutable_block: {self.num_blocks} \t {self.e.item():.3f}, \t'
            f'self.taylor: \t{self.taylor.min().item():.3f}\t{self.taylor.max().item():.3f}\t'  # noqa
            f'mask:\t{get_mask_str()}\t')

    # inherit from BaseMutable

    @property
    def current_choice(self):
        return super().current_choice

    def fix_chosen(self, chosen):
        return super().fix_chosen(chosen)

    def dump_chosen(self):
        return super().dump_chosen()


class MutableHead(BaseMutable, DMSMutableMixIn):

    def __init__(self, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self._dms_mutable_mixin_init(num_heads)

        mask = torch.ones([num_heads])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor

        self.flop_scale_converter = None

    @property
    def current_imp(self):
        if self.flop_scale_converter is None:
            return super().current_imp
        else:
            return self.flop_scale_converter(super().current_imp)

    @property
    def current_imp_flop(self):
        if self.flop_scale_converter is None:
            return super().current_imp_flop
        else:
            return self.flop_scale_converter(super().current_imp_flop)

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def current_choice(self):
        return None

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 0, 1.0)

    def info(self):

        def get_mask_str():
            mask_str = ''
            for i in range(self.num_heads):
                if self.mask[i] == 1:
                    mask_str += '1'
                else:
                    mask_str += '0'
            return mask_str

        return super().info() + f'\t{get_mask_str()}\t'


class MutableChannelForHead(BaseMutable, DMSMutableMixIn):

    def __init__(self, num_channels, num_heads) -> None:
        super().__init__()
        self._dms_mutable_mixin_init(num_channels)
        self.num_head = num_heads
        self.num_channels = num_channels

        self.taylor = self.taylor.reshape([num_heads, -1])

        mask = torch.ones([num_channels])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor
        self.mask = self.mask.reshape([num_heads, -1])

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def current_choice(self):
        return None


class MutableChannelWithHead(SimpleMutableChannel):

    def __init__(self, mutable_head: MutableHead,
                 mutable_channel: MutableChannelForHead) -> None:
        super().__init__(mutable_channel.num_channels)

        self.mutable_head = mutable_head
        self.mutable_channel = mutable_channel

    @property
    def current_imp(self):
        channel_imp = self.mutable_channel.current_imp
        head_imp = self.mutable_head.current_imp
        imp = head_imp.unsqueeze(-1) * channel_imp
        imp = imp.flatten()
        return imp

    @property
    def current_imp_flop(self):
        current_imp_flop = self.mutable_channel.current_imp_flop
        head_imp = self.mutable_head.current_imp_flop
        imp = head_imp.unsqueeze(-1) * current_imp_flop
        imp = imp.flatten()
        return imp

    @property
    def current_mask(self):
        channel = self.mutable_channel.mask
        head = self.mutable_head.mask.unsqueeze(-1)

        return (channel * head).bool().flatten()

    @torch.no_grad()
    def limit_value(self):
        self.mutable_head.limit_value()
        self.mutable_channel.limit_value()

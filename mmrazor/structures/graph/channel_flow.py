# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import List, Union


class BaseChannelUnit:
    """BaseChannelUnit is a collection of BaseChannel.

    All  BaseChannels are saved in two lists: self.input_related and
    self.output_related.
    """

    def __init__(self) -> None:

        self.input_related: List[BaseChannel] = []
        self.output_related: List[BaseChannel] = []

    # collect units


class BaseChannel:
    pass


class ChannelElem:

    def __init__(self) -> None:
        self._parent: Union[None, 'ChannelElem'] = None

    # channel elem operations

    @classmethod
    def union_two(cls, elem1: 'ChannelElem', elem2: 'ChannelElem'):
        if elem1.root is not elem2.root:
            elem2.root._set_parent(elem1)

    def union(self, elem: 'ChannelElem'):
        ChannelElem.union_two(self, elem)

    # work as a disjoint set

    @property
    def root(self) -> 'ChannelElem':
        if self._parent is None:
            return self
        else:
            return self._parent.root

    def _set_parent(self, parent: 'ChannelElem'):
        assert self._parent is None
        self._parent = parent


class ChannelTensor:

    def __init__(self, num_channel_elem: int) -> None:
        self.elems = [ChannelElem() for _ in range(num_channel_elem)]

    # tensor operations

    def union(self, tensor: 'ChannelTensor'):
        return self.__class__.union_two(self, tensor)

    @classmethod
    def union_two(cls, tensor1: 'ChannelTensor', tensor2: 'ChannelTensor'):
        assert len(tensor1) == len(tensor2)
        for e1, e2 in zip(tensor1, tensor2):
            ChannelElem.union_two(e1, e2)

    @classmethod
    def cat(cls, tensors: List['ChannelTensor']):
        elems = list(itertools.chain(*[t.elems for t in tensors]))
        new_tensor = ChannelTensor(len(elems))
        new_tensor.elems = elems
        return new_tensor

    def expand(self, expand_ratio: int):
        new_tensor = ChannelTensor(expand_ratio * len(self))

        for i in range(len(self)):
            for j in range(expand_ratio):
                new_tensor[i * expand_ratio + j].union(self[i])
        return new_tensor

    # work as a tensor

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            return self.elems[key]
        elif isinstance(key, slice):
            elems = self.elems[key]
            tensor = ChannelTensor(len(elems))
            tensor.elems = elems
            return tensor
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.elems)

    def __iter__(self):
        for e in self.elems:
            yield e

    def __add__(self, tensor: 'ChannelTensor'):
        return ChannelTensor.cat([self, tensor])

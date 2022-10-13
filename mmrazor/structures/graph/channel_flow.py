# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
from typing import List, Set, Union

from mmrazor.utils import IndexDict


class ChannelElem:

    def __init__(self, owning_tensor, index_in_tensor) -> None:
        self._parent: Union[None, 'ChannelElem'] = None
        self._subs: Set[ChannelElem] = set()
        self.owing_tensor = owning_tensor
        self.index_in_tensoor = index_in_tensor

    # channel elem operations

    @classmethod
    def union_two(cls, elem1: 'ChannelElem', elem2: 'ChannelElem'):
        if elem1.root is not elem2.root:
            elem2.root._set_parent(elem1)

    def union(self, elem: 'ChannelElem'):
        ChannelElem.union_two(self, elem)

    # unit related

    @property
    def owing_elem_set(self):
        root = self.root
        return root.subs

    @property
    def elem_set_hash(self):
        tensor_list = list(self.owing_elem_set)
        tensor_set = set([elem.owing_tensor for elem in tensor_list])
        frozen_set = frozenset(tensor_set)
        return frozen_set.__hash__()

    # work as a disjoint set

    @property
    def root(self) -> 'ChannelElem':
        if self._parent is None:
            return self
        else:
            return self._parent.root

    @property
    def subs(self):
        subs = copy.copy(self._subs)
        subs.add(self)
        for elem in self._subs:
            subs = subs.union(elem.subs)
        return subs

    def _set_parent(self, parent: 'ChannelElem'):
        assert self._parent is None
        self._parent = parent
        parent._subs.add(self)


class ChannelTensor:

    def __init__(self, num_channel_elem: int) -> None:
        self.elems = [ChannelElem(self, i) for i in range(num_channel_elem)]

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

    # unit operation

    @property
    def elems_hash(self):
        elem_hashes = [elem.elem_set_hash for elem in self.elems]
        return elem_hashes

    @property
    def elems_hash_dict(self):
        elem_hashes = self.elems_hash
        unit_dict = IndexDict()
        start = 0
        for e in range(1, len(self)):
            if elem_hashes[e] != elem_hashes[e - 1]:
                unit_dict[(start, e)] = elem_hashes[start]
                start = e
        unit_dict[start, len(self)] = elem_hashes[start]
        return unit_dict

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

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
        self._hash_cache = None
        self._min_elem_set_index_cache = None

    # channel elem operations

    @classmethod
    def union_two(cls, elem1: 'ChannelElem', elem2: 'ChannelElem'):
        root1 = elem1.root
        root2 = elem2.root
        if root1 is not root2:
            root2._set_parent(root1)

    def union(self, elem: 'ChannelElem'):
        ChannelElem.union_two(self, elem)

    # unit related

    @property
    def owing_elem_set(self):
        root = self.root
        return root.subs

    def reset_cache(self):
        self._hash_cache = None
        self._min_elem_set_index_cache = None

    @property
    def elem_set_hash(self):
        if self._hash_cache is not None:
            return self._hash_cache
        else:
            tensor_list = list(self.owing_elem_set)
            tensor_set = set([elem.owing_tensor for elem in tensor_list])
            frozen_set = frozenset(tensor_set)
            hash = frozen_set.__hash__()
            for elem in self.owing_elem_set:
                assert elem._hash_cache is None
                elem._hash_cache = hash
            return hash

    @property
    def min_elem_set_index(self):
        if self._min_elem_set_index_cache is not None:
            return self._min_elem_set_index_cache
        else:
            elem_set = self.owing_elem_set
            min_index = int(pow(2, 32))
            for elem in elem_set:
                min_index = min(min_index, elem.index_in_tensoor)
            for elem in elem_set:
                assert elem._min_elem_set_index_cache is None
                elem._min_elem_set_index_cache = min_index
            return min_index

    # work as a disjoint set

    @property
    def root(self) -> 'ChannelElem':
        if self._parent is None:
            return self
        else:
            root = self._parent.root
            self._unset_parent()
            self._set_parent(root)
            return root

    @property
    def subs(self):
        subs = copy.copy(self._subs)
        subs.add(self)
        for elem in self._subs:
            subs = subs.union(elem.subs)
        return subs

    def _set_parent(self, parent: 'ChannelElem'):
        assert self._parent is None
        assert parent.root is not self
        self._parent = parent
        parent._subs.add(self)

    def _unset_parent(self):
        assert self._parent is not None
        old_parent = self._parent
        old_parent._subs.remove(self)
        self._parent = None


class ChannelTensor:

    def __init__(self, num_channel_elem: int) -> None:
        self.elems = [ChannelElem(self, i) for i in range(num_channel_elem)]

    # tensor operations

    def union(self, tensor: 'ChannelTensor'):
        return self.__class__.union_two(self, tensor)

    @classmethod
    def union_two(cls, tensor1: 'ChannelTensor', tensor2: 'ChannelTensor'):
        assert len(tensor1) == len(tensor2), f'{len(tensor1)}!={len(tensor2)}'
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
                self[i].union(new_tensor[i * expand_ratio + j])
        return new_tensor

    # unit operation

    @property
    def elems_hash_with_index(self):
        elem_hashes = [(elem.elem_set_hash, elem.min_elem_set_index)
                       for elem in self.elems]
        return elem_hashes

    @property
    def elems_hash_dict(self):
        elem_hash_with_index = self.elems_hash_with_index
        unit_dict = IndexDict()
        start = 0
        for e in range(1, len(self)):
            if (elem_hash_with_index[e][0] != elem_hash_with_index[e - 1][0]
                    or elem_hash_with_index[e][1] <
                    elem_hash_with_index[e - 1][1]):

                unit_dict[(start, e)] = elem_hash_with_index[start][0]
                start = e
        unit_dict[start, len(self)] = elem_hash_with_index[start][0]
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

    def _reset_channel_elem_cache(self):
        for elem in self.elems:
            elem.reset_cache()

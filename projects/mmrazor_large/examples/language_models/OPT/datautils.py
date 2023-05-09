import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DistributedSampler


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-train.arrow'
    )
    testdata = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-test.arrow'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/ptb_text_only/penn_treebank/1.1.0/8d1b97746fb9765d140e569ec5ddd35e20af4d37761f5e1bf357ea0b081f2c1f/ptb_text_only-train.arrow'
    )
    testdata = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/ptb_text_only/penn_treebank/1.1.0/8d1b97746fb9765d140e569ec5ddd35e20af4d37761f5e1bf357ea0b081f2c1f/ptb_text_only-validation.arrow'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(' '.join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(' '.join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=True
    # )
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',use_auth_token=True
    # )
    traindata0 = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/allenai___json/allenai--c4-6fbe877195f42de5/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-train-00000-of-00002.arrow'
    )
    traindata1 = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/allenai___json/allenai--c4-6fbe877195f42de5/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-train-00001-of-00002.arrow'
    )
    traindata = concatenate_datasets([traindata0, traindata1])
    valdata = Dataset.from_file(
        '/nvme/liukai/.cache/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-validation.arrow'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)


def fold_tokens(tokens: torch.Tensor, batch_seq_len=2048):
    # tokens: 1 N
    N = tokens.shape[1]
    num_drop = N % batch_seq_len
    if num_drop != 0:
        tokens = tokens[:, :-num_drop]
    tokens = tokens.reshape([-1, batch_seq_len])  # B N
    return tokens


class LanguageDataset(TorchDataset):

    def __init__(self, seq: torch.Tensor, seq_len: int = 2048) -> None:
        super().__init__()
        # seq: 1, N
        self.seq_len = seq_len

        self.seq = fold_tokens(seq)  # B N

    def __len__(self) -> int:
        return self.seq.shape[0]

    def __getitem__(self, index):
        return self.seq[index]


def build_language_loader(testloader, world_size, rank, model, batch_size=128):
    val_dataset = LanguageDataset(testloader.input_ids, seq_len=model.seqlen)
    distributed_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    batch_size = min(len(val_dataset) // world_size, batch_size)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=distributed_sampler)
    return val_dataloader

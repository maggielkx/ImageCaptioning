import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from readStory import Vocabulary
import json as js


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.vist = js.load(open(json, 'r'))
        self.ids = list(self.vist.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vist = self.vist
        vocab = self.vocab
        ann_id = self.ids[index]
        captions = vist[ann_id]['sents']
        img_ids = vist[ann_id]['image_ids']
        split = vist[ann_id]['split']
        images =[]
        targets=[]
        infos=[]
        for i, (caption, img_id) in enumerate(zip(captions, img_ids)):

            # Convert caption (string) to word ids.
            tokens = nltk.tokenize.word_tokenize(caption)
            caption = ['<start>'] + tokens + ['<end>']

            sentence = ' '.join(caption)
            info = [img_id, split, sentence]
            caption = [vocab(token) for token in caption]
            target = torch.Tensor(caption)

            image = Image.open(os.path.join(self.root, split, img_id)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
            targets.append(target)
            infos.append(info)

        return images, targets, infos

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
            - info: [imagename, caption]

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """

    flat_data = [item for sublist in data for item in sublist]

    # remove info from data tuple
    infos = [x[-1] for x in flat_data]
    data = [x[:-1] for x in flat_data]

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader




import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from dataLoader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, EncoderRNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn.functional as F


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def pairwise_ranking_loss(margin, h_s, h_i_pos, h_i_negatives):
    """
    Pair-wise rank loss (Frome 2013) for sentence and image encoders to map to one common multimodal space

    h_s.data.shape = (N, D)
    h_i_pos.data.shape = (N, D)
    h_i_neg.data.shape = (N, D)

    """

    N = h_s.data.shape[0]  # args.batch_size
    h_s = F.normalize(h_s, p=2)
    h_i_pos = F.normalize(h_i_pos, p=2)

    score_pos = torch.sum((h_s * h_i_pos), 1)
    loss = 0.0

    for h_i_neg in h_i_negatives:
        h_i_neg = F.normalize(h_i_neg, p=2)
        score_neg = torch.sum((h_s * h_i_neg), 1)
        loss += torch.clamp(margin + score_neg - score_pos, min=0, max=100000000000.0)
    loss = torch.sum(loss) / float(N)
    return loss


def encoder_preprocess(images, captions, lengths):

    images = to_var(images)
    captions = to_var(captions)
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    return images, captions, targets


def train_loss_calc(images, captions, lengths, targets, criterion, encoder, decoder, encoder_retrieval):

    with torch.no_grad():
        features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, targets)

    # retrieval model
    features_retrieval = encoder_retrieval(captions)
    mask = features_retrieval.clone()
    mask[mask != 0] = 1
    features_retrieval = features_retrieval * mask.cuda()
    features_retrieval = torch.sum(features_retrieval, 1) / features_retrieval.shape[1]

    # calculating pair ranking loss
    margin = 1
    features_negatives = []
    for k in range(args.neg_samples):
        index = np.random.permutation(features.shape[0])
        features_neg = features[index]
        features_negatives.append(features_neg)

    rank_loss = pairwise_ranking_loss(margin, features_retrieval, features, features_negatives)

    loss_final = rank_loss + loss

    return loss_final


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    train_loader = get_loader(args.train_image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.valid_image_dir, args.caption_path, vocab,
                                   transform, args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size)
    encoder_retrieval = EncoderRNN(args.embed_size, args.hidden_size,
                                   len(vocab), dropout=0.1)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab, args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        encoder_retrieval.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + list(
        encoder_retrieval.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # params_retrieval = list(encoder_retrieval.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # optimizer_retrieval = torch.optim.Adam(params_retrieval, lr=args.learning_rate)

    # Train the Models
    total_step = len(train_loader)
    loss_v_best = 0
    for epoch in range(args.num_epochs):
        for i, (images_t, captions_t, lengths_t), (images_v, captions_v, lengths_v) in enumerate(zip(train_loader, val_loader)):

            # Set mini-batch dataset
            images_t, captions_t, targets_t = encoder_preprocess(images_t, captions_t, lengths_t)
            images_v, captions_v, targets_v = encoder_preprocess(images_v, captions_v, lengths_v)

            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            encoder_retrieval.zero_grad()

            loss_t = train_loss_calc(images_t, captions_t, lengths_t, targets_t, criterion, encoder, decoder, encoder_retrieval)
            loss_v = train_loss_calc(images_v, captions_v, lengths_v, targets_v, criterion, encoder, decoder, encoder_retrieval)

            loss_t.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], TrainLoss: %.4f, ValidLoss: %.4f' %
                      (epoch, args.num_epochs, i, total_step, loss_t.data, loss_v.data))

            # Save the models
            if loss_v > loss_v_best:

                loss_v_best = loss_v
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder_retrieval.state_dict(),
                           os.path.join(args.model_path,
                                        'retriev-encoder-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_5.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='./resizedSotry/train',
                        help='directory for resized train images')
    parser.add_argument('--valid_image_dir', type=str, default='./resizedSotry/val',
                        help='directory for resized valid images')
    parser.add_argument('--caption_path', type=str,
                        default='vist.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--neg_samples', type=int, default=5)
    args = parser.parse_args()
    print(args)
    main(args)

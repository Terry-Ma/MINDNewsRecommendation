import torch
import numpy as np
import logging
import yaml
import os
import random

from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from torchtext.vocab import Vocab

PAD = '<pad>'
UNK = '<unk>'
NewsPAD = 'NewsPAD'
data_path = '../../data/'
logger = logging.getLogger()

def load_news(news_file, data_set, use_abstract, use_body, news_max_len):
    logger.info('load news, dataset: {}'.format(data_set))
    nid2nidx = {}
    nidx2words = []
    cur_nidx = 0
    train_news_path = '{}/data_{}/train/{}'.format(data_path, data_set, news_file)
    val_news_path = '{}/data_{}/val/{}'.format(data_path, data_set, news_file)
    test_news_path = '{}/test/{}'.format(data_path, news_file)
    nid2nidx, nidx2words, cur_nidx = \
        load_news_file(train_news_path, use_abstract, use_body, news_max_len, nid2nidx, nidx2words, cur_nidx)
    train_nidx = cur_nidx
    logger.info('train set news num {}'.format(train_nidx))
    for path in (val_news_path, test_news_path):
        nid2nidx, nidx2words, cur_nidx = \
            load_news_file(path, use_abstract, use_body, news_max_len, nid2nidx, nidx2words, cur_nidx)
    logger.info('total news num {}'.format(cur_nidx))
    # add NewsPAD
    nid2nidx[NewsPAD] = cur_nidx
    nidx2words.append([UNK] + [PAD] * (news_max_len - 1))
    
    return nid2nidx, nidx2words, train_nidx

def load_news_file(news_path, use_abstract, use_body, news_max_len, nid2nidx, nidx2words, cur_nidx):
    with open(news_path, encoding='utf-8') as f:
        for line in f:
            nid, _, _, title, abstract, _, _, _, body = line.replace('\n', '').split('\t')
            if nid not in nid2nidx:
                nid2nidx[nid] = cur_nidx
                text = title
                if use_abstract:
                    text += ' '
                    text += abstract
                if use_body:
                    text += ' '
                    text += body
                text_split = text.split()
                nidx2words.append(text_split[:news_max_len] if len(text_split) >= news_max_len else\
                    text_split + [PAD] * (news_max_len - len(text_split)))
                cur_nidx += 1
    
    return nid2nidx, nidx2words, cur_nidx

def generate_vocab(nidx2words, train_nidx, word_min_freq, vocab_path):
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
        logger.info('load existing vocab {}, vocab_size {}'.format(vocab_path, len(vocab)))
    else:
        word2freq = defaultdict(int)
        for nidx in range(train_nidx):
            for word in nidx2words[nidx]:
                word2freq[word] += 1
        if PAD in word2freq:
            del word2freq[PAD]
        vocab = Vocab(Counter(word2freq), min_freq=word_min_freq, specials=[UNK, PAD])
        torch.save(vocab, vocab_path)
        logger.info('generate vocab and save: {}, vocab size {}'.format(vocab_path, len(vocab)))
    
    return vocab

def transform_words(vocab, nidx2words):
    nidx2widxes = [[vocab.stoi[word] for word in words] for words in nidx2words]
    nidx2mask = [[int(widx == vocab.stoi[PAD]) for widx in widxes] for widxes in nidx2widxes]
    nidx2widxes = torch.tensor(nidx2widxes)
    nidx2mask = torch.tensor(nidx2mask)

    return nidx2widxes, nidx2mask

def load_behavior(nid2nidx, hist_max_len, data_set, neg_pos_ratio, batch_size):
    train_behavior_path = '{}/data_{}/train/behaviors.tsv'.format(data_path, data_set)
    val_behavior_path = '{}/data_{}/val/behaviors.tsv'.format(data_path, data_set)
    test_behavior_path = '{}/test/behaviors.tsv'.format(data_path)
    # train
    train_uidx2nidxes, train_uidx2mask, train_iter, _ = load_behavior_file(
        train_behavior_path, hist_max_len, 'train', neg_pos_ratio, nid2nidx, batch_size)
    # val
    val_uidx2nidxes, val_uidx2mask, val_iter, _ = load_behavior_file(
        val_behavior_path, hist_max_len, 'val', neg_pos_ratio, nid2nidx, batch_size)
    # test
    test_uidx2nidxes, test_uidx2mask, test_iter, test_iid2num = load_behavior_file(
        test_behavior_path, hist_max_len, 'test', neg_pos_ratio, nid2nidx, batch_size)

    return train_uidx2nidxes, train_uidx2mask, train_iter, val_uidx2nidxes, val_uidx2mask, val_iter, \
        test_uidx2nidxes, test_uidx2mask, test_iter, test_iid2num

def load_behavior_file(behavior_path, hist_max_len, data_type, neg_pos_ratio, nid2nidx, batch_size):
    assert data_type in ('train', 'val', 'test')

    uid2uidx = {}
    uidx2nids = []
    uidxes = []
    can_nids = []
    labels = []
    iid2num = [0]  # impr_id start from 1
    cur_uidx = 0
    with open(behavior_path, encoding='utf-8') as f:
        for line in f:
            _, uid, _, hist_nids, impr_nids = line.replace('\n', '').split('\t')
            if uid not in uid2uidx:
                uid2uidx[uid] = cur_uidx
                hist_nids = hist_nids.split()
                uidx2nids.append(hist_nids[:hist_max_len] if len(hist_nids) >= hist_max_len else\
                    hist_nids + [NewsPAD] * (hist_max_len - len(hist_nids)))
                cur_uidx += 1
            if data_type != 'train':
                num = 0
                for impr_nid in impr_nids.split():
                    if data_type == 'val':
                        labels.append(int(impr_nid[-1]))
                    uidxes.append(uid2uidx[uid])
                    can_nids.append(impr_nid.split('-')[0])
                    num += 1
                if data_type == 'test':
                    iid2num.append(num)
            else:   # apply negative sampling to training data
                pos_nids = []
                neg_nids = []
                for impr_nid in impr_nids.split():
                    if int(impr_nid[-1]):
                        pos_nids.append(impr_nid.split('-')[0])
                    else:
                        neg_nids.append(impr_nid.split('-')[0])
                can_nids += pos_nids
                uidxes += [uid2uidx[uid]] * len(pos_nids)
                labels += [1] * len(pos_nids)
                neg_nids_num = int(len(pos_nids) * neg_pos_ratio)
                neg_nid_idxes = random.choices(list(range(len(neg_nids))), k=neg_nids_num)
                for idx in neg_nid_idxes:
                    can_nids.append(neg_nids[idx])
                    uidxes.append(uid2uidx[uid])
                    labels.append(0)
    can_nidxes, uidx2nidxes, uidx2mask = transform_nids(nid2nidx, can_nids, uidx2nids)
    uidxes = torch.tensor(uidxes)
    labels = torch.tensor(labels)
    dataset = torch.utils.data.TensorDataset(can_nidxes, uidxes, labels) if data_type != 'test' else\
        torch.utils.data.TensorDataset(can_nidxes, uidxes)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True) if data_type == 'train' else\
        torch.utils.data.DataLoader(dataset, batch_size)
    logger.info('load behavior file {}, data type {}, sample num {}'.\
        format(behavior_path, data_type, len(uidxes)))

    return uidx2nidxes, uidx2mask, data_iter, iid2num

def transform_nids(nid2nidx, can_nids, uidx2nids):
    can_nidxes = [nid2nidx[nid] for nid in can_nids]
    uidx2nidxes = [[nid2nidx[nid] for nid in nids] for nids in uidx2nids]
    uidx2mask = [[int(nidx == nid2nidx[NewsPAD]) for nidx in nidxes] for nidxes in uidx2nidxes]
    can_nidxes = torch.tensor(can_nidxes)
    uidx2nidxes = torch.tensor(uidx2nidxes)
    uidx2mask = torch.tensor(uidx2mask)

    return can_nidxes, uidx2nidxes, uidx2mask

if __name__ == '__main__':
    data_path = '../data/'
    nid2nidx, _, _ = load_news('news_clean.tsv', 'large', 0, 0, 16)
    test_uidx2nidxes, test_uidx2mask, test_iter, test_iid2num = load_behavior_file(
        '../data/test/behaviors.tsv', 32, 'test', 3, nid2nidx, 512)
    print(test_iid2num[:5])
    print(len(test_iid2num))
    print(sum(test_iid2num))
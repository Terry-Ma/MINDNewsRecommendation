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
data_path = '../data/'
logger = logging.getLogger()

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

def load_news(news_file, data_set, use_abstract, use_body, news_max_len):
    logger.info('load news, dataset: {}'.format(data_set))
    nid2nidx = {}
    nidx2words = []
    cur_nidx = 0
    train_news_path = '{}/data_{}/train/{}'.format(data_path, data_set, news_file)
    val_news_path = '{}/data_{}/val/{}'.format(data_path, data_set, news_file)
    test_news_path = '{}/test/{}'.format(data_path, news_file)
    cur_nidx, nid2nidx, nidx2words = \
        load_news_file(train_news_path, use_abstract, use_body, news_max_len, nid2nidx, nidx2words, cur_nidx)
    train_nidx = cur_nidx
    logger.info('train set news num {}'.format(train_nidx))
    for path in (val_news_path, test_news_path):
        cur_nidx, nid2nidx, nidx2words = \
            load_news_file(path, use_abstract, use_body, news_max_len, nid2nidx, nidx2words, cur_nidx)
    logger.info('total news num {}'.format(cur_nidx))
    # add NewsPAD
    nid2nidx[NewsPAD] = cur_nidx
    nidx2words.append([UNK] + [PAD] * (news_max_len - 1))
    
    return nid2nidx, nidx2words, train_nidx

def generate_vocab(nidx2words, train_nidx, word_min_freq, vocab_path):
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
        logger.info('load existing vocab {}'.format(vocab_path))
    else:
        word2freq = defaultdict(int)
        for nidx in range(train_nidx):
            for word in nidx2words[nidx]:
                word2freq[word] += 1
        if PAD in word2freq:
            del word2freq[PAD]
        vocab = Vocab(Counter(word2freq), min_freq=word_min_freq, specials=[UNK, PAD])
        torch.save(vocab, vocab_path)
        logger.info('generate vocab and save: {}'.format(vocab_path))
    
    return vocab

def transform_words(vocab, nidx2words):
    nidx2widxes = [[vocab.stoi[word] for word in words] for words in nidx2words]
    nidx2mask = [[int(widx == vocab.stoi[PAD]) for widx in widxes] for widxes in nidx2widxes]
    nidx2widxes = torch.tensor(nidx2widxes)
    nidx2mask = torch.tensor(nidx2mask)

    return nidx2widxes, nidx2mask

''' torch iter
can_nidx: (batch_size)
user_hist_nidx: (batch_size, hist_seq_len)
label: (batch_size)
'''

''' nrms iter
can_news: (batch_size, news_seq_len)
can_word_mask: (batch_size, news_seq_len)
user_hist_news: (batch_size, hist_seq_len, news_seq_len)
user_hist_word_mask: (batch_size, hist_seq_len, news_seq_len)
user_hist_news_mask: (batch_size, hist_seq_len)
'''

def load_behavior_file(behavior_path, hist_max_len, data_type, neg_pos_ratio):
    assert data_type in ('train', 'val', 'test')

    uid2uidx = {}
    uidx2nids = []
    uidxs = []
    can_nidxes = []
    labels = []
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
                for impr_nid in impr_nids.split():
                    if data_type == 'val':
                        labels.append(int(impr_nid[-1]))
                    uidxs.append(uid2uidx[uid])
                    can_nidxes.append(impr_nid.split('-')[0])
            else:   # apply negative sampling to training data
                pos_nids = []
                neg_nids = []
                for impr_nid in impr_nids.split():
                    if int(impr_nid[-1]):
                        pos_nids.append(impr_nid.split('-')[0])
                    else:
                        neg_nids.append(impr_nid.split('-')[0])
                can_nidxes += pos_nids
                uidxs += [uid2uidx[uid]] * len(pos_nids)
                labels += [1] * len(pos_nids)
                neg_nids_num = int(len(pos_nids) * neg_pos_ratio)
                neg_nid_idxs = random.choices(list(range(len(neg_nids))), k=neg_nids_num)
                for idx in neg_nid_idxs:
                    can_nidxes.append(neg_nids[idx])
                    uidxs.append(uid2uidx[uid])
                    labels.append(0)
    logger.info('load behavior file {}, data type {}, sample num {}'.\
        format(behavior_path, data_type, len(uidxs)))

    return uidx2nids, can_nidxes, uidxs, labels

def transform_nids(nid2nidx, uidx2nids):
    uidx2nidxes = [[nid2nidx[nid] for nid in nids] for nids in uidx2nids]
    uidx2mask = [[int(nidx == nid2nidx[NewsPAD]) for nidx in nidxes] for nidxes in uidx2nidxes]

    return uidx2nidxes, uidx2mask
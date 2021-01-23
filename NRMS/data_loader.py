import logging
import sys
sys.path.append('../')

from data_preprocess.preprocess import *

logger = logging.getLogger()

class NRMSDataLoader:
    def __init__(self, config):
        self.config = config
        self.nid2nidx, self.nidx2words, self.train_nidx = load_news(
            news_file=config['preprocess']['news_file'],
            data_set=config['preprocess']['data_set'],
            use_abstract=config['preprocess']['use_abstrace'],
            use_body=config['preprocess']['use_body'],
            news_max_len=config['preprocess']['news_max_len']
            )
        self.vocab = generate_vocab(
            nidx2words=self.nidx2words,
            train_nidx=self.train_nidx,
            word_min_freq=config['preprocess']['word_min_freq'],
            vocab_path=config['preprocess']['vocab_path']
            )
        self.nidx2widxes, self.nidx2mask = transform_words(
            vocab=self.vocab,
            nidx2words=self.nidx2words
            )
        self.train_uidx2nidxes, self.train_uidx2mask, self.train_iter, self.val_uidx2nidxes, self.val_uidx2mask, \
            self.val_iter, self.test_uidx2nidxes, self.test_uidx2mask, self.test_iter = load_behavior(\
                nid2nidx=self.nid2nidx,
                hist_max_len=config['preprocess']['hist_max_len'],
                data_set=config['preprocess']['data_set'],
                neg_pos_ratio=config['preprocess']['neg_pos_ratio'],
                batch_size=config['preprocess']['batch_size']
                ) 

    def iter_with_label(self, batch_iter, uidx2nidxes, uidx2mask):
        for can_nidxes, uidxes, labels in batch_iter:
            can_news = self.nidx2widxes[can_nidxes]  # (batch_size, news_seq_len)
            can_word_mask = self.nidx2mask[can_nidxes]  # (batch_size, news_seq_len)
            user_hist_nidxes = uidx2nidxes[uidxes]  # (batch_size, hist_seq_len)
            # (batch_size, hist_seq_len, news_seq_len)
            user_hist_news = self.nidx2widxes[user_hist_nidxes.view(-1)].view(\
                self.config['preprocess']['batch_size'], -1, self.config['preprocess']['news_max_len'])
            # (batch_size, hist_seq_len, news_seq_len)
            user_hist_word_mask = self.nidx2mask[user_hist_nidxes.view(-1)].view(\
                self.config['preprocess']['batch_size'], -1, self.config['preprocess']['news_max_len'])
            user_hist_news_mask = uidx2mask[uidxes]  # (batch_size, hist_seq_len)

            yield (
                can_news,
                can_word_mask,
                user_hist_news,
                user_hist_word_mask,
                user_hist_news_mask,
                labels
                )
    
    def iter_without_label(self, batch_iter, uidx2nidxes, uidx2mask):
        for can_nidxes, uidxes in batch_iter:
            can_news = self.nidx2widxes[can_nidxes]  # (batch_size, news_seq_len)
            can_word_mask = self.nidx2mask[can_nidxes]  # (batch_size, news_seq_len)
            user_hist_nidxes = uidx2nidxes[uidxes]  # (batch_size, hist_seq_len)
            # (batch_size, hist_seq_len, news_seq_len)
            user_hist_news = self.nidx2widxes[user_hist_nidxes.view(-1)].view(\
                self.config['preprocess']['batch_size'], -1, self.config['preprocess']['news_max_len'])
            # (batch_size, hist_seq_len, news_seq_len)
            user_hist_word_mask = self.nidx2mask[user_hist_nidxes.view(-1)].view(\
                self.config['preprocess']['batch_size'], -1, self.config['preprocess']['news_max_len'])
            user_hist_news_mask = uidx2mask[uidxes]  # (batch_size, hist_seq_len)

            yield (
                can_news,
                can_word_mask,
                user_hist_news,
                user_hist_word_mask,
                user_hist_news_mask
                )
    
    def train_batch_iter(self):
        return self.iter_with_label(self.train_iter, self.train_uidx2nidxes, self.train_uidx2mask)
    
    def val_batch_iter(self):
        return self.iter_with_label(self.val_iter, self.val_uidx2nidxes, self.val_uidx2mask)
    
    def test_batch_iter(self):
        return self.iter_without_label(self.test_iter, self.test_uidx2nidxes, self.test_uidx2mask)
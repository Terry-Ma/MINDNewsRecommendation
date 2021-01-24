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
            use_abstract=config['preprocess']['use_abstract'],
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
                batch_size=config['train']['batch_size']
                )

    def iter_with_label(self, batch_iter, uidx2nidxes, uidx2mask):
        for can_nidxes, uidxes, labels in batch_iter:
            can_news, can_word_mask = self.process_nidxes(can_nidxes)
            user_hist_news, user_hist_word_mask, user_hist_news_mask = \
                self.process_uidxes(uidxes, uidx2nidxes, uidx2mask)

            yield {
                'can_news': can_news,  # (batch_size, news_seq_len)
                'can_word_mask': can_word_mask,  # (batch_size, news_seq_len)
                'user_hist_news': user_hist_news,  # (batch_size, hist_seq_len, news_seq_len)
                'user_hist_word_mask': user_hist_word_mask,  # (batch_size, hist_seq_len, news_seq_len)
                'user_hist_news_mask': user_hist_news_mask,  # (batch_size, hist_seq_len)
                'labels': labels
                }
    
    def iter_without_label(self, batch_iter, uidx2nidxes, uidx2mask):
        for can_nidxes, uidxes in batch_iter:
            can_news, can_word_mask = self.process_nidxes(can_nidxes)
            user_hist_news, user_hist_word_mask, user_hist_news_mask = \
                self.process_uidxes(uidxes, uidx2nidxes, uidx2mask)

            yield {
                'can_news': can_news,
                'can_word_mask': can_word_mask,
                'user_hist_news': user_hist_news,
                'user_hist_word_mask': user_hist_word_mask,
                'user_hist_news_mask': user_hist_news_mask
                }

    def generate_embedding(self, model):
        # generate nidx2embed & uidx2embed
        batch_size = self.config['train']['batch_size']
        nid_num = self.nid2nidx.shape[0]
        val_uid_num = self.val_uidx2nidxes.shape[0]
        test_uid_num = self.test_uidx2nidxes.shape[0]
        self.nidx2embed = torch.rand(nid_num, self.config['model']['hidden_size'])
        self.val_uidx2embed = torch.rand(val_uid_num, self.config['model']['hidden_size'])
        self.test_uidx2embed = torch.rand(test_uid_num, self.config['model']['hidden_size'])
        # nidx2embed
        for nidx in range(0, nid_num, batch_size):
            nidxes = torch.tensor(list(range(nidx, min(nidx + batch_size, nid_num + 1))))
            news, word_mask = self.process_nidxes(nidxes)
            self.nidx2embed[nidx:min(nidx + batch_size, nid_num + 1), :] = \
                model.news2embed(news, word_mask).to('cpu')
        # uidx2embed
        for uidx2embed, uidx2nidxes, uidx2mask in zip(
            [self.val_uidx2embed, self.test_uidx2embed],
            [self.val_uidx2nidxes, self.test_uidx2nidxes],
            [self.val_uidx2mask, self.test_uidx2mask]
            ):
            uid_num = uidx2embed.shape[0]
            for uidx in range(0, uid_num, batch_size):
                uidxes = torch.tensor(list(range(uidx, min(uidx + batch_size, uid_num + 1))))
                user_hist_news, user_hist_word_mask, user_hist_news_mask = \
                    self.process_uidxes(uidxes, uidx2nidxes, uidx2mask)
                uidx2embed[uidx:min(uidx + batch_size, uid_num + 1), :] = \
                    model.user2embed(user_hist_news, user_hist_word_mask, user_hist_news_mask).to('cpu')
    
    def process_nidxes(self, nidxes):
        news = self.nidx2widxes[nidxes]  # (batch_size, news_seq_len)
        word_mask = self.nidx2mask[nidxes]  # (batch_size, news_seq_len)

        return news, word_mask

    def process_uidxes(self, uidxes, uidx2nidxes, uidx2mask):
        user_hist_nidxes = uidx2nidxes[uidxes]  # (batch_size, hist_seq_len)
        # (batch_size, hist_seq_len, news_seq_len)
        user_hist_news = self.nidx2widxes[user_hist_nidxes.view(-1)].view(\
            self.config['train']['batch_size'], -1, self.config['preprocess']['news_max_len'])
        # (batch_size, hist_seq_len, news_seq_len)
        user_hist_word_mask = self.nidx2mask[user_hist_nidxes.view(-1)].view(\
            self.config['train']['batch_size'], -1, self.config['preprocess']['news_max_len'])
        user_hist_news_mask = uidx2mask[uidxes]  # (batch_size, hist_seq_len)

        return user_hist_news, user_hist_word_mask, user_hist_news_mask
    
    def val_inference(self):
        for can_nidxes, uidxes, labels in self.val_iter:
            yield self.batch_inference(can_nidxes, uidxes), labels

    def test_inference(self):
        for can_nidxes, uidxes in self.test_iter:
            yield self.batch_inference(can_nidxes, uidxes)

    def batch_inference(self, nidxes, uidxes):
        news_embed = self.nidx2embed[nidxes]
        user_embed = self.val_uidx2embed[uidxes]
        scores = (news_embed * user_embed).sum(dim=1)  # (batch_size)
        
        return scores
    
    def train_batch_iter(self):
        return self.iter_with_label(self.train_iter, self.train_uidx2nidxes, self.train_uidx2mask)
    
    def val_batch_iter(self):
        return self.iter_with_label(self.val_iter, self.val_uidx2nidxes, self.val_uidx2mask)
    
    def test_batch_iter(self):
        return self.iter_without_label(self.test_iter, self.test_uidx2nidxes, self.test_uidx2mask)
    
    def vocab_size(self):
        return len(self.vocab)
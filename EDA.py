import numpy as np

from collections import defaultdict

def get_vocab(news_path):
    word2freq = defaultdict(int)
    with open(news_path, encoding='utf-8') as f:
        for line in f:
            _, _, _, title, abstract, _, _, _, body = line.replace('\n', '').split('\t')
            for word in title.split():
                word2freq[word] += 1
            for word in abstract.split():
                word2freq[word] += 1
            for word in body.split():
                word2freq[word] += 1
    
    return word2freq

def vocab_size(news_file):
    train_vocab = get_vocab('./data/data_large/train/{}'.format(news_file))
    test_vocab = get_vocab('./data/test/{}'.format(news_file))
    print('file {}'.format(news_file))
    print('train vocab size {}, test vocab size {}'.\
        format(len(train_vocab), len(test_vocab)))
    for i in range(10, 110, 10):
        print('> {}, train vocab size {}, test vocab size {}'.format(
            i,
            len([k for k in train_vocab if train_vocab[k] > i]),
            len([k for k in test_vocab if test_vocab[k] > i])
            ))

def get_text_length(news_path):
    title_lens = []
    abstract_lens = []
    body_lens = []
    with open(news_path, encoding='utf-8') as f:
        for line in f:
            _, _, _, title, abstract, _, _, _, body = line.replace('\n', '').split('\t')
            title_lens.append(len(title.split()))
            abstract_lens.append(len(abstract.split()))
            body_lens.append(len(body.split()))
    title_lens = np.array(title_lens)
    abstract_lens = np.array(abstract_lens)
    body_lens = np.array(body_lens)
    concat_lens = title_lens + abstract_lens + body_lens

    return title_lens, abstract_lens, body_lens, concat_lens

def text_length(news_file):
    train_title_lens, train_abstract_lens, train_body_lens, train_concat_lens = \
        get_text_length('./data/data_large/train/{}'.format(news_file))
    test_title_lens, test_abstract_lens, test_body_lens, test_concat_lens = \
        get_text_length('./data/test/{}'.format(news_file))
    print('file {}'.format(news_file))
    for i in range(50, 110, 10):
        print('{}% - train: title {}, abstract {}, body {}, concat {}; test: title {}, abstract {}, body {}, concat {}'.format(
            i,
            np.percentile(train_title_lens, i),
            np.percentile(train_abstract_lens, i),
            np.percentile(train_body_lens, i),
            np.percentile(train_concat_lens, i),
            np.percentile(test_title_lens, i),
            np.percentile(test_abstract_lens, i),
            np.percentile(test_body_lens, i),
            np.percentile(test_concat_lens, i)
            ))

def get_hist_news_num(impr_path):
    hist_news_nums = []
    impr_news_nums = []
    impr_news_pos_nums = []
    with open(impr_path, encoding='utf-8') as f:
        for line in f:
            _, _, _, hist_news, impr_news = line.replace('\n', '').split('\t')
            hist_news_nums.append(len(hist_news.split()))
            impr_news_split = impr_news.split()
            impr_news_nums.append(len(impr_news_split))
            impr_news_pos_nums.append(0 if 'test' in impr_path else sum([int(i[-1]) for i in impr_news_split]))
    hist_news_nums = np.array(hist_news_nums)
    impr_news_nums = np.array(impr_news_nums)
    impr_news_pos_nums = np.array(impr_news_pos_nums)

    return hist_news_nums, impr_news_nums, impr_news_pos_nums

def hist_impr_news_num():
    train_hist_news_nums, train_impr_news_nums, train_impr_news_pos_nums = \
        get_hist_news_num('./data/data_large/train/behaviors.tsv')
    val_hist_news_nums, val_impr_news_nums, val_impr_news_pos_nums = \
        get_hist_news_num('./data/data_large/val/behaviors.tsv')
    test_hist_news_nums, test_impr_news_nums, test_impr_news_pos_nums = \
        get_hist_news_num('./data/test/behaviors.tsv')
    print('train total sample num {}, val total sample num {}, test total sample num {}'.format(
        train_impr_news_nums.sum(),
        val_impr_news_nums.sum(),
        test_impr_news_nums.sum()
        ))
    for i in range(50, 110, 10):
        print('{}%, train: hist num {}, impr nums {}, impr pos nums {}; test: hist num {}, impr nums {}'.format(
            i,
            np.percentile(train_hist_news_nums, i),
            np.percentile(train_impr_news_nums, i),
            np.percentile(train_impr_news_pos_nums, i),
            np.percentile(test_hist_news_nums, i),
            np.percentile(test_impr_news_nums, i)
            ))

def get_hist_news(impr_path):
    uid2nids = {}
    with open(impr_path, encoding='utf-8') as f:
        for line in f:
            _, uid, _, hist_news, _ = line.replace('\n', '').split('\t')
            if uid not in uid2nids:
                uid2nids[uid] = set([hist_news])
            else:
                uid2nids[uid].add(hist_news)
    
    return uid2nids

def hist_news_change():
    train_uid2nids = get_hist_news('./data/data_large/train/behaviors.tsv')
    test_uid2nids = get_hist_news('./data/test/behaviors.tsv')
    print('train: user num {}, hist not change num {}'.format(
        len(train_uid2nids),
        len([uid for uid in train_uid2nids if len(train_uid2nids[uid]) == 1])
        ))
    print('test: user num {}, hist not change num {}'.format(
        len(test_uid2nids),
        len([uid for uid in test_uid2nids if len(test_uid2nids[uid]) == 1])
        ))

def impr_id_change():
    with open('./data/test/behaviors.tsv', encoding='utf-8') as f:
        pre_impr_id = 0
        pre_uid = 0
        impr_id_mistake_num = 0
        uid_mistake_num = 0
        for line in f:
            impr_id, uid, _, _, _ = line.replace('\n', '').split('\t')
            impr_id = int(impr_id)
            if impr_id != pre_impr_id + 1:
                impr_id_mistake_num += 1
            if uid == pre_uid:
                uid_mistake_num += 1
            pre_uid = uid
            pre_impr_id = impr_id
    print('impr id mistake num {}, uid mistake num {}'.format(impr_id_mistake_num, uid_mistake_num))

if __name__ == '__main__':
    # news_files = [
    #     'news_clean.tsv',
    #     'news_bpe.tsv',
    #     'news_bpe_20000.tsv'
    #     ]
    # for news_file in news_files:
        # vocab_size(news_file)
        # text_length(news_file)
    
    # hist_impr_news_num()

    # hist_news_change()

    impr_id_change()
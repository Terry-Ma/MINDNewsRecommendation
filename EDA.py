from collections import defaultdict

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

if __name__ == '__main__':
    news_files = [
        'news_clean.tsv',
        'news_bpe.tsv',
        'news_bpe_20000.tsv'
        ]
    for news_file in news_files:
        vocab_size(news_file)        
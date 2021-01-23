import os

def text_clean(
    paths, 
    news_file, 
    news_clean_file
    ):
    ''' lower case
    '''
    for path in paths:
        with open('{}/{}'.format(path, news_file), encoding='utf-8') as f,\
            open('{}/{}'.format(path, news_clean_file), mode='w', encoding='utf-8') as g:
            for line in f:
                line_split = line.replace('\n', '').split('\t')
                # title-3 abstract-4 body-8
                for i in [3, 4, 8]:
                    line_split[i] = line_split[i].lower()
                g.write('\t'.join(line_split) + '\n')

def generate_bpe_corpus(
    paths, 
    news_clean_file, 
    title_file, 
    abstract_file, 
    body_file,
    news_bpe_corpus_file
    ):
    ''' prepare bpe train & test corpus
    '''
    for path in paths:
        with open('{}/{}'.format(path, news_clean_file), encoding='utf-8') as f,\
            open('{}/{}'.format(path, title_file), mode='w', encoding='utf-8') as t,\
            open('{}/{}'.format(path, abstract_file), mode='w', encoding='utf-8') as a,\
            open('{}/{}'.format(path, body_file), mode='w', encoding='utf-8') as b:
            for line in f:
                _, _, _, title, abstract, _, _, _, body = line.replace('\n', '').split('\t')
                t.write(title + '\n')
                a.write(abstract + '\n')
                b.write(body + '\n')
    news_bpe_corpus_path = '{}/{}'.format(paths[0], news_bpe_corpus_file)
    with open(news_bpe_corpus_path, encoding='utf-8', mode='w') as f:
        for corpus_file in [title_file, abstract_file, body_file]:
            with open('{}/{}'.format(paths[0], corpus_file), encoding='utf-8') as g:
                for line in g:
                    f.write(line)

def learn_apply_bpe(
    paths,
    news_bpe_corpus_file,
    title_file,
    abstract_file,
    body_file,
    title_bpe_file,
    abstract_bpe_file,
    body_bpe_file,
    bpe_code_file,
    num_operations
    ):
    ''' train on news_bpe_corpus_file and apply to title & abstract & body
    '''
    news_bpe_corpus_path = '{}/{}'.format(paths[0], news_bpe_corpus_file)
    bpe_code_path = '{}/{}'.format(paths[0], bpe_code_file)
    os.system('subword-nmt learn-bpe -s {} < {} > {}'.\
        format(num_operations, news_bpe_corpus_path, bpe_code_path))
    for path in paths:
        for from_file, to_file in zip(
            [title_file, abstract_file, body_file],
            [title_bpe_file, abstract_bpe_file, body_bpe_file]
            ):
            bpe_from_path = '{}/{}'.format(path, from_file)
            bpe_to_path = '{}/{}'.format(path, to_file)
            os.system('subword-nmt apply-bpe -c {} < {} > {}'.\
                format(bpe_code_path, bpe_from_path, bpe_to_path))

def generate_news_bpe(
    paths,
    news_clean_file,
    title_bpe_file,
    abstract_bpe_file,
    body_bpe_file,
    news_bpe_file
    ):
    ''' merge title_bpe & abstract_bpe & body_bpe with other news info
    '''
    for path in paths:
        bpe_texts = []
        for text_file in [title_bpe_file, abstract_bpe_file, body_bpe_file]:
            with open('{}/{}'.format(path, text_file), encoding='utf-8') as f:
                bpe_texts.append(f.readlines())
        with open('{}/{}'.format(path, news_clean_file), encoding='utf-8') as f,\
            open('{}/{}'.format(path, news_bpe_file), mode='w', encoding='utf-8') as g:
            for i, line in enumerate(f):
                line_split = line.replace('\n', '').split('\t')
                line_split[3] = bpe_texts[0][i].replace('\n', '')
                line_split[4] = bpe_texts[1][i].replace('\n', '')
                line_split[8] = bpe_texts[2][i].replace('\n', '')
                g.write('\t'.join(line_split) + '\n')

def prepare_data():
    paths = [
        '../data/data_large/train/',   # large_train for lean_apply_bpe
        '../data/data_large/val/',
        '../data/data_small/train/',
        '../data/data_small/val/',
        '../data/data_demo/train/',
        '../data/data_demo/val/',
        '../data/test/'
    ]
    news_file = 'news_with_body.tsv' 
    news_clean_file = 'news_clean.tsv'
    title_file = 'title'
    abstract_file = 'abstract'
    body_file = 'body'
    news_bpe_corpus_file = 'news_bpe_corpus'
    num_operations = 20000
    title_bpe_file = 'title_bpe_{}'.format(num_operations)
    abstract_bpe_file = 'abstract_bpe_{}'.format(num_operations)
    body_bpe_file = 'body_bpe_{}'.format(num_operations)
    bpe_code_file = 'bpe_code_{}'.format(num_operations)
    news_bpe_file = 'news_bpe_{}.tsv'.format(num_operations)
    
    text_clean(
        paths, 
        news_file, 
        news_clean_file
        )
    generate_bpe_corpus(
        paths, 
        news_clean_file, 
        title_file, 
        abstract_file, 
        body_file,
        news_bpe_corpus_file
        )
    learn_apply_bpe(
        paths,
        news_bpe_corpus_file,
        title_file,
        abstract_file,
        body_file,
        title_bpe_file,
        abstract_bpe_file,
        body_bpe_file,
        bpe_code_file,
        num_operations
        )
    generate_news_bpe(
        paths,
        news_clean_file,
        title_bpe_file,
        abstract_bpe_file,
        body_bpe_file,
        news_bpe_file
        )

if __name__ == '__main__':
    prepare_data()
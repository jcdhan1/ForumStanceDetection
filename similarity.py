import os, reader, writer,random
import numpy as np
from gensim import models, similarities, corpora
from twokenize_wrapper.twokenize import tokenize

def concatenate_posts(topic, rdr):
    #Order of posts within a debate matter but the order of the debates do not
    #Posts are concatenated in order but debates are concatenated randomly
    debates_for_target = []
    for prefix in reader.subsetAZ(dir_cd + topic):
        dbt = rdr.load_cd(unseen_target, prefix)
        bodies = [p.body for p in dbt.post_list]
        doc = concatenate(bodies)
        debates_for_target += [doc]
    random.shuffle(debates_for_target)
    return debates_for_target

def concatenate(string_list):
    return ' '.join(string_list)

if __name__ == '__main__':
    dir_cd = './data/CreateDebate/'
    trials = []
    for trial in range(5):
        rdr=reader.Reader(dir_cd)
        
        seen_target_doc = concatenate(concatenate_posts('obama', rdr))
        tokenised_seen_target_doc = writer.filterStopwords(tokenize(seen_target_doc.lower()))
        rdr.dir_lst.remove('obama')
        
        documents = []
        for unseen_target in rdr.dir_lst:
            print('concatenating posts for', unseen_target)
            unseen_target_debates = concatenate_posts(unseen_target, rdr)
            unseen_target_doc = concatenate(unseen_target_debates)
            tokenised_debate = writer.filterStopwords(tokenize(unseen_target_doc.lower()))
            documents += [tokenised_debate]
        
        dictionary = corpora.Dictionary(documents)
        dictionary.save(os.path.join('./results/unseen_targets.dict'))
        dictionary = corpora.Dictionary.load('./results/unseen_targets.dict')
        
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        corpora.MmCorpus.serialize(os.path.join('./results/unseen_targets.mm'), corpus)
        corpus = corpora.MmCorpus('./results/unseen_targets.mm')
        
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=3)
    
        vec_bow = dictionary.doc2bow(tokenised_seen_target_doc)
        vec_lsi = lsi[vec_bow]
        
        index = similarities.MatrixSimilarity(lsi[corpus])
        
        index.save('./results/unseen_targets.index')
        index = similarities.MatrixSimilarity.load('./results/unseen_targets.index')
        
        similarities = index[vec_lsi]
        
        print(list(enumerate(similarities)))
        
        similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
    
        trials += [similarities]
    print(np.array(trials))
    np.save('./results/topic_similarities_to_obama.npy', np.array(trials))
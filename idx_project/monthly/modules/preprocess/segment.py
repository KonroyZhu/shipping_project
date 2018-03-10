import jieba

def load_data(fliename):
    root="../../data/"
    with open(root+fliename) as f:
        data=f.readlines()
    return data

def segment(filename):
    '''
    segment process for sentences
    :param filename: path for the file which save the sentences data
    :return: list of segmented sentences
    '''
    stopWords=load_data("stopWords")
    sentenceList=load_data(filename)
    sentenceSegment=[]
    for sentence in sentenceList:
        wordList=jieba.cut(sentence,cut_all=False)
        wordList2=[]
        for w in wordList:
            if w not in stopWords and w != '':
                wordList2.append(w)

        sentenceSegment.append(list(wordList2))
    return list(sentenceSegment)

def tagSegment(tags):
    '''
    segment process for tags
    :param filename: path for the file which save the tags
    :return: list of segmented tags
    '''
    tagList = []
    for tag in tags:
        if tag == "\n" or tag == "":
            tagList.append(["\n"])
        else:
            items = []
            T = tag.split(" ")
            for t in T:
                try:
                    mark = t.split("#")[1]
                    t= t.split("#")[0]
                except:
                    pass
                # t = t.replace('\n', '')
                item = " ".join(jieba.cut(t))
                items.append(item+"#"+mark)
            tagList.append(items)
    return tagList

def save(segSentence,name):
    with open('../../data/'+name,'w') as f:
        for seg in segSentence:
            for w in seg:
                f.write(w+" ")



if __name__ == '__main__':
    '''
    # segment for sentences
    segSentence=segment("sentences.txt")
    print(len(segSentence))
    save(segSentence,"sentenceSeg")
    '''

    '''
    # segment for tags
    tags=load_data("tag")
    tagList=tagSegment(tags)
    save(tagList,"tagSeg")
    '''



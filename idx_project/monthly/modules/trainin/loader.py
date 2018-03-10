class Loader:
    def __init__(self):
        self.root="../../data/"
        self.sentencePath="../../data/sentenceSeg"
        self.tagPath="../../data/tag"

    def load(self,name):
        List = open(self.root+name).readlines()
        return List
    
    def loadSentence(self):
        sentenceList = open(self.sentencePath).readlines()

        reformedList = []
        for sentence in sentenceList:
            wordList = sentence.split(" ")
            reformedList.append(wordList)
        return reformedList

    def loadTag(self):
        tagList = open(self.tagPath).readlines()

        reformedList = []
        for tag in tagList:
            tag = tag.replace("\n", " ").strip()
            tags = tag.split(" ")
            if tags[0] != '':
                reformedList.append(tags)
            else:
                reformedList.append([])
        return reformedList
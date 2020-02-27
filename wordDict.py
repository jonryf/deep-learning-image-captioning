import pickle
import nltk
from collections import defaultdict
from pycocotools.coco import COCO

class WordDict:
    def __init(self):
        self.indexToWord = {}
        self.wordToIndex = {}
        self.addWordToDict('<pad>')
        self.addWordToDict('<start>')
        self.addWordToDict('<end>')
        self.addWordToDict('<unk>')
        
    def clear(self):
        self.indexToWord = {}
        self.wordToIndex = {}
        self.addWordToDict('<pad>')
        self.addWordToDict('<start>')
        self.addWordToDict('<end>')
        self.addWordToDict('<unk>')
    
    def getIndexForWord(self, word):
        if word not in self.wordToIndex:
            return self.wordToIndex["<unk>"]
        return self.wordToIndex[word]

    def getWordForIndex(self, index):
        if index not in self.indexToWord:
            return "<unk>"
        return self.indexToWord[index]
    
    def __call__(self, item):
        if type(item) is str:
            return self.getIndexForWord(item)
        if type(item) is int:
            return self.getWordForIndex(item)
        return None

    def addWordToDict(self, word):
        if word not in self.wordToIndex:
            curIdx = len(self.wordToIndex)
            self.indexToWord[curIdx] = word
            self.wordToIndex[word] = curIdx

    def buildWordDictFromWords(self, words):
        for word in words:
            self.addWordToDict(word)
            
    def buildWordDictFromJson(self, json, threshold):
        nltk.download('punkt')
        coco = COCO(json)
        ids = coco.anns.keys()
        numberOccurences = defaultdict(lambda: 0)
        print("loading dictionary..........")
        for id in ids:
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            for token in tokens:
                numberOccurences[token] += 1

        allWords = []
        for word in numberOccurences:
            if numberOccurences[word] >= threshold:
                allWords.append(word)

        self.clear()
        self.buildWordDictFromWords(allWords)
        print("finished loading dictionary")
        
        
def getWordDict(json, threshold, savePath="wordDictFile"):
    savePath += "threshold" + str(threshold)
    try:
        f = open(savePath, "rb")
        ret = pickle.load(f)
        print("Loaded wordDict from file %s" % savePath)
        return ret
    except:
        print("Could not load wordDict from file %s. Loading from json" % savePath)
        
    ret = WordDict()
    ret.buildWordDictFromJson(json, threshold)
    f = open(savePath, "wb")
    pickle.dump(ret,f)
    return ret

import numpy as np


def loadGlove():

    print("Loading glove")
    print("--------------")
    words = []
    idx = 0
    word2idx = {}
    weights = []

    count = 0


    with open(f'./glove.6B.50d.txt', 'rb') as doc:
        for line in doc:
            line = line.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            weight = np.array(line[1:]).astype(np.float)
            weights.append(weight)
            count += 1
            if(count % 50000 == 0):
                print("{}/400000".format(count))
    
    glove = {word: weights[word2idx[word]] for word in words}
    return glove

if __name__ == "__main__":
    loadGlove()

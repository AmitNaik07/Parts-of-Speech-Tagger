import nltk
import tkFileDialog
import matplotlib.pyplot as plt; plt.rcdefaults() 
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from gtts import gTTS
from Tkinter import *
from nltk import pos_tag, word_tokenize

tagged_sentences =nltk.corpus.treebank.tagged_sents()

def features(sentence, index):                        #used to define the features on which words should be classified.
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

cutoff = int(.70 * len(tagged_sentences))                     #taking 70% for training and 30% for testing
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]
 
print len(training_sentences)   
print len(test_sentences)         
 
def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y
 
X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])
 
clf.fit(X[:10000], y[:10000])   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
 
print 'Training completed'
 
X_test, y_test = transform_to_dataset(test_sentences)
 
print "Accuracy:", clf.score(X_test, y_test)

def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)

def callback():
    s =E1.get()
    tts = gTTS(text=s, lang='en')
    tts.save("good.mp3")
    os.system("cvlc good.mp3")

def explorer():
    fname =tkFileDialog.askopenfilename(filetypes = (("Template files", "*.type"),("All files", "*")))
    file =open(fname, 'r')
    rd =file.read()
    tts =gTTS(text = rd, lang ='en')
    os.system("cvlc good.mp3")

def getPOS():
    fname =tkFileDialog.askopenfilename(filetypes = (("All files", "*"),("Template files", "*.type")))
    file =open(fname, 'r')
    rd =file.read()
    text =word_tokenize(rd)

    tagged =pos_tag(text)
    print tagged
    length =len(tagged) -1

    noun,adjective,verb =([] for i in range(3))

    for i in tagged:
        if i[1][0] == 'N':
            noun.append(i[0])
        elif i[1][0] == 'J':
            adjective.append(i[0])
        elif i[1][0] == 'V':
            verb.append(i[0])

    data =np.array([len(noun), len(adjective), len(verb)])
    names =('Noun', 'Adjective', 'Verb')
    y_pos =np.arange(len(names))
    plt.bar(y_pos, data, align ='center', alpha =0.5)
    plt.xticks(y_pos, names, rotation =45)
    plt.xlabel('Parts of Speech')
    plt.ylabel('Frequency')
    plt.title("Parts of Speech Occurence")
    plt.show()                                       #plotting frequency of noun,verb,adjective on bar graph


root =Tk()
frame = Frame(root, width=200, height=200)        #Tkinkter window for GUI
root.wm_title("Text to Speech")

L1 = Label(root, text="Enter text")
L1.pack( side = TOP)
E1 = Entry(root, bd =5, highlightcolor ='yellow')
E1.pack(side = TOP)

b = Button(root, text="Play", width=10, command=callback)
b.pack()
f = Button(root, text="Browse", width=10, command=explorer)
f.pack()
n = Button(root, text="POS Tagger", width=10, command=getPOS)
n.pack()
root.mainloop()
root.quit()

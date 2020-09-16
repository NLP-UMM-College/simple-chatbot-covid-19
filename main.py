import requests 
import nltk
import numpy
import tflearn
import tensorflow as tf
import random
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup 
from urllib.request import urlopen
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

kumpulan_pertanyaan = ["hello","hai","yuhuu","halo","Assalamualaikum"]
kumpulan_jawaban    = ["hola","baik","hai","halo","Waalikumsalam"]
j = 0
jawaban = ""
temukan_jawaban = False
jawaban_selesai = False
baris_jawaban   = False

with open("COVID-Dialogue-Dataset-English.txt","rt",encoding = "utf8",errors="ignore") as anriswa:
   for myline in anriswa:
        myline = myline.lower()
        for i in myline :
            if i == '?'  :
                temukan_jawaban = False                    
                break     
        if temukan_jawaban :
            jawaban += myline
        if not temukan_jawaban and baris_jawaban:
            kumpulan_jawaban.append(jawaban)
            jawaban = ""
            jawaban_selesai = False
            baris_jawaban   = False
        for i in myline :
            if i == '?':
                kata = ""
                for i in myline:
                    kata += i
                    if i == '?':
                        break                         
                kumpulan_pertanyaan.append(kata)
                temukan_jawaban = True
                baris_jawaban = True
                break                    


kumpulan_pertanyaan_dan_jawaban = dict(zip(kumpulan_pertanyaan,kumpulan_jawaban))     

kumpulan_kata = []
docs_x        = []
docs_y        = []

for pertanyaan in kumpulan_pertanyaan_dan_jawaban:
    kata_kata = nltk.word_tokenize(pertanyaan)
    kumpulan_kata.extend(kata_kata)
    docs_x.append(kata_kata)
    docs_y.append(kumpulan_pertanyaan_dan_jawaban[pertanyaan])

labels = kumpulan_jawaban

factory = StemmerFactory()
stemmer = factory.create_stemmer()
kumpulan_kata = [stemmer.stem(kata.lower()) for kata in kumpulan_kata if kata != "?"]
kumpulan_kata = sorted(list(set(kumpulan_kata)))

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in kumpulan_kata:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

# print(len(output[0]))

# Kalau ingin pakai K-Fold

# accuracies = []

# for fold in range(1, 11):
#     tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    # adam = tflearn.optimizers.Adam(learning_rate=0.0001)
net = tflearn.regression(net, optimizer='adam', metric='accuracy')


model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='/tmp/tflearn_logs/')
# model.load("model/test")
model.fit(training, output, n_epoch=1500, batch_size=16, show_metric=True)
score = model.evaluate(training, output)
    # accuracies.append(score[0])
print(model.evaluate(training, output))
    # print(classification_report(training, output))
    # model.save("model/batch8epoch1000")
model.save('model/test_v2_16sept_1500epoch')

# accuracies = numpy.array(accuracies)
# print(accuracies)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
        
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("user \t\t: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, kumpulan_kata)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        print("Roboragi \t: ",tag)
        
chat()


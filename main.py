from random import shuffle

import copy
from sklearn import neighbors, datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier

#atributos de teste
attributes = [
    {'buying': ['vhigh', 'high', 'med', 'low']},
    {'maint': ['vhigh', 'high', 'med', 'low']},
    {'doors': ['2', '3', '4', '5more']},
    {'persons': ['2', '4', 'more']},
    {'lug_boot': ['small', 'med', 'big']},
    {'safety': ['low', 'med', 'high']}
]

#abre o arquivo
f = open('car.data.txt')

#lista de items
items = []

#itera sobre o arquivo
for line in f:
    #faz split dos parametros
    split_line = line.split(',')
    #adiciona a um dicionario
    items.append({'buying': split_line[0], 'maint': split_line[1], 'doors': split_line[2],
                  'persons': split_line[3], 'lug_boot': split_line[4], 'safety': split_line[5],
                  'original_classification': split_line[6]})

f.close()

def getItems(rawItems):
    items = copy.deepcopy(rawItems)
    for item in items:
        item.pop('original_classification')
    return items

def getClassifications(rawItems):
    items = copy.deepcopy(rawItems)
    values = []
    for item in items:
        values.append({'original_classification':(item.pop('original_classification'))})
    return values
    

#randomiza a lista
shuffle(items)
#ponto em que se deve dividir
middle_point = int(len(items) * 0.8)
#set de 80% para treinamento
training_set = items[:middle_point]
#set de 20% para classificação
classification_set = items[middle_point:]

print(items)
print(training_set)
print(classification_set)


trainingXVectorizer = DictVectorizer(sparse=False)
trainingYVectorizer = DictVectorizer(sparse=False)
classificationXVectorizer = DictVectorizer(sparse=False)

clf = neighbors.KNeighborsClassifier()
clf.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
knnresult = (clf.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
print(trainingYVectorizer.inverse_transform(knnresult))

mlp = MLPClassifier()
mlp.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
mlpresult = (mlp.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
print(trainingYVectorizer.inverse_transform(mlpresult))
print(getClassifications(classification_set))




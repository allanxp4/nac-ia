from random import shuffle
from sklearn import neighbors, datasets
from sklearn.feature_extraction import DictVectorizer

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
    items = rawItems.copy()
    for item in items:
        items.pop('original_classification')
    return items

def getClassifications(rawItems)
    items = rawItems.copy()
    for k in rawItems.keys():
        if(k != 'original_classification'):
            del items[k]
    return items
    



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

trainingXVectorizer = DictVectorizer()
trainingYVectorizer = DictVectorizer()
classificationYVectorizer = DictVectorizer()

clf = neighbors.KNeighborsClassifier()
clf.fit(trainingXVectorizer.fit_transform(getItems(training_set), trainingYVectorizer(getClassifications(training_set))
print(clf.predict(classificationYVectorizer.fit_transform(classification_set)))














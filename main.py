from random import shuffle

import copy
from sklearn import neighbors, datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

#atributos de teste
#http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
attributes = [
    {'buying': ['vhigh', 'high', 'med', 'low']},
    {'maint': ['vhigh', 'high', 'med', 'low']},
    {'doors': ['2', '3', '4', '5more']},
    {'persons': ['2', '4', 'more']},
    {'lug_boot': ['small', 'med', 'big']},
    {'safety': ['low', 'med', 'high']},
    {'classification': ['unacc', 'acc', 'good', 'v_good']}
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

#metodos helpers
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

def getKeyValueString(items):
    results = []
    for item in items:
        for key,value in item.items():
            results.append(key + "=" + value)
    return results

def getArrayOfDict(dictList):
    results = []
    for item in dictList:
        for key, value in item.items():
            results.append(key )
    return results

def getDifference(list1, list2):
    for i in len(list1):
        print(list1[i])


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

#vetoriza os dados
#a normalização é apliada automaticamente por padrão
trainingXVectorizer = DictVectorizer(sparse=False)
trainingYVectorizer = DictVectorizer(sparse=False)
classificationXVectorizer = DictVectorizer(sparse=False)

#resultados da lista de terinamento
originalValues = getKeyValueString(getClassifications(classification_set))

#knn k=1
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
k1result = (clf.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
k1prettyresult = (trainingYVectorizer.inverse_transform(k1result))

#knn k=3
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
k3result = (clf.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
k3prettyresult = (trainingYVectorizer.inverse_transform(k3result))

#knn k=5
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
k5result = (clf.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
k5prettyresult = (trainingYVectorizer.inverse_transform(k5result))

#mlp camadas intermediarias = 1
mlp = MLPClassifier(hidden_layer_sizes=1)
mlp.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
mlp1result = (mlp.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
mlp1prettyresult = (trainingYVectorizer.inverse_transform(mlp1result))

#mlp camadas intermediarias = 2
mlp = MLPClassifier(hidden_layer_sizes=2)
mlp.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
mlp2result = (mlp.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
mlp2prettyresult = (trainingYVectorizer.inverse_transform(mlp2result))

#mlp camadas intermediarias = 3
mlp = MLPClassifier(hidden_layer_sizes=3)
mlp.fit(trainingXVectorizer.fit_transform(getItems(training_set)), trainingYVectorizer.fit_transform(getClassifications(training_set)))
mlp3result = (mlp.predict(classificationXVectorizer.fit_transform(getItems(classification_set))))
mlp3prettyresult = (trainingYVectorizer.inverse_transform(mlp3result))

#pega as diferenças entre o resultado esperado e o predito
mlp1difference = (set(getArrayOfDict(mlp1prettyresult)) & set(getKeyValueString(getClassifications(classification_set))))
mlp2difference = (set(getArrayOfDict(mlp2prettyresult)) & set(getKeyValueString(getClassifications(classification_set))))
mlp3difference = (set(getArrayOfDict(mlp3prettyresult)) & set(getKeyValueString(getClassifications(classification_set))))
k1difference = (set(getArrayOfDict(k1prettyresult)) & set(getKeyValueString(getClassifications(classification_set))))
k3difference = (set(getArrayOfDict(k3prettyresult)) & set(getKeyValueString(getClassifications(classification_set))))
k5difference = (set(getArrayOfDict(k5prettyresult)) & set(getKeyValueString(getClassifications(classification_set))))

#checa a precisão
mlp1precision = len(mlp1difference) / len(classification_set)*100
mlp2precision = len(mlp2difference) / len(classification_set)*100
mlp3precision = len(mlp3difference) / len(classification_set)*100
k1precision = len(k1difference) / len(classification_set)*100
k3precision = len(k3difference) / len(classification_set)*100
k5precision = len(k5difference) / len(classification_set)*100

print(mlp1precision)
print(mlp2precision)
print(mlp3precision)
print(k1precision)
print(k3precision)
print(k5precision)

objects = ("mlp l=1", "mlp l=2", "mlp l=3", "knn k=1", "knn k=3", "knn k=5")
y_pos = np.arange(len(objects))
labels = [mlp1precision, mlp2precision, mlp3precision, k1precision, k3precision, k5precision]

plt.bar(y_pos, labels, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Taxa de erro')
plt.xlabel('Método usado')
plt.title('Precisão de diferentes métodos com o dataset Car Evaluation')

plt.show()
#percebi que esse dataset não tem exatamente um nivel muito grande de variabilidade, então, não há uma differença muito grande de precisão dependendo
#da lista que foi randomizada, mas, mostraram uma taxa de acerto bem alta.








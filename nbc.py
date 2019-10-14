import csv
import numpy as np
import pandas as pd
import re
import math
import sys


def preProcessCsv(filename):
  line = pd.read_csv(filename, sep=',', quotechar='"', header=0, engine='python')
  matrix = line.as_matrix()

  #create column for multivalue columns
  ambience = [] #10
  parking=[] #11
  diet = [] #12
  recommend = [] #16

  #counting unique values in multivalue columns
  for row in matrix:
    if row[10] is not np.nan:
      target = row[10]
      pattern = r'[\w]+'
      found = re.findall(pattern, target)
      for n in found:
        if n in ambience:
          continue
        else:
          ambience.append(n)

    if row[11] is not np.nan:
      target = row[11]
      pattern = r'[\w]+'
      found = re.findall(pattern, target)
      for n in found:
        if n in parking:
          continue
        else:
          parking.append(n)

    if row[12] is not np.nan:
      target = row[12]
      pattern = r'[\w]+'
      found = re.findall(pattern, target)
      for n in found:
        if n in diet:
          continue
        else:
          diet.append(n)

    if row[16] is not np.nan:
      target = row[16]
      pattern = r'[\w]+'
      found = re.findall(pattern, target)
      for n in found:
        if n in recommend:
          continue
        else:
          recommend.append(n)

  # create a shifter to compensate for added columns
  shifter = 0

  # check for each value in multivalue columns
  # and create a new column for it
  for attr in ambience:
    a = []
    shifter+=1
    for row in line['ambience']:
      if row is not np.nan:
        if attr in row:
          a.append(1)
        else:
          a.append(0)
      else:
        a.append(0)
    line.insert(10+shifter, attr, a)

  for attr in parking:
    a = []
    shifter+=1
    for row in line['parking']:
      if row is not np.nan:
        if attr in row:
          a.append(1)
        else:
          a.append(0)
      else:
        a.append(0)
    line.insert(11+shifter, attr, a)

  for attr in diet:
    a = []
    for row in line['dietaryRestrictions']:
      if row is not np.nan:
        if attr in row:
          a.append(1)
        else:
          a.append(0)
      else:
        a.append(0)
    line.insert(12+shifter, attr, a)

  for attr in recommend:
    a = []
    for row in line['recommendedFor']:
      if row is not np.nan:
        if attr in row:
          a.append(1)
        else:
          a.append(0)
      else:
        a.append(0)
    line.insert(16+shifter, attr, a)

  # delete the original multivalue columns
  line = line.drop('ambience',axis=1)
  line = line.drop('parking',axis=1)
  line = line.drop('dietaryRestrictions',axis=1)
  line = line.drop('recommendedFor',axis=1)

  # replace empty value with none
  line = line.replace(np.nan, 999)
  dataset = line.as_matrix()

  #print((dataset[290][8]))

  #faced some problem when compare 'nan' in int column
  for i in range(len(dataset)):
    for j in range(len(dataset[0])):
      if dataset[i][j] == 999:
        dataset[i][j] = None

  return dataset


def nbcTrain(dataset):
  classT = 0.0
  classF = 0.0

  #datasetT = dataset.T

  # print(datasetT[8][290])

  for r in dataset[:,-1]:
    if r == 1:
      classT += 1
    elif r == 0:
      classF += 1

  tList = []
  fList = []

  # calculate the number of unique values in each column
  process = pd.DataFrame(dataset)
  for i in process:
    if i == 43:
      continue
    uniqueList =  process[i].unique()
    countList = [0] * len(uniqueList)
    listLen = len(uniqueList)

    tDict = {}
    fDict = {}

    #count numbers of each unique values
    for j in process[i]:
      for k in range(listLen):
        if j == uniqueList[k]:
          countList[k] += 1

    #count numbers of T and F for each unique value
    for j in range(listLen):
      numT = 0.0
      numF = 0.0
      for k in range(len(dataset)):
        if (dataset[k][i] == uniqueList[j])&(dataset[k][-1]==1):
          numT += 1
        elif (dataset[k][i] == uniqueList[j])&(dataset[k][-1]==0):
          numF += 1
      
      #smooth probability
      pT = (numT + 1) / (classT + listLen)
      pF = (numF + 1) / (classF + listLen)

      # #without smoothing
      # pT = (numT) / (classT)
      # pF = (numF) / (classF)

      tDict[uniqueList[j]] = pT
      fDict[uniqueList[j]] = pF

    tList.append(tDict)
    fList.append(fDict)

  pClass = (classT+1) / ((classT+classF)+2)


  return tList, fList, pClass


def nbcPredict(dataset, tList, fList, pClass):

  resultList=[]
  tProbList = []

  for row in dataset:
    #create likelihood for T and F
    lT = math.log(pClass)
    lF = math.log(1-pClass)
    for i in range(len(row)-1):
      avgT = 0.0
      avgF = 0.0

      for valF in tList[i].items():
        avgT += valF[1]
      for valT in fList[i].items():
        avgF += valT[1]


      avgT = avgT/len(tList[i])
      avgF = avgF/len(fList[i])
      
      if row[i] in tList[i]:
        lT += math.log(tList[i][row[i]])
      else:
        lT += math.log(avgT)

      if row[i] in fList[i]:
        lF += math.log(fList[i][row[i]])
      else:
        lF += math.log(avgF)

    if lT > lF:
      resultList.append(1)
    else:
      resultList.append(0)

    #taking log inverse
    lT = math.exp(lT)
    lF = math.exp(lF)

    #calculate the pi for squared loss function
    pT = lT/(lT+lF)
    tProbList.append(pT)
    

  return resultList, tProbList

if __name__ == "__main__":
  train = preProcessCsv(sys.argv[1])
  test = preProcessCsv(sys.argv[2])
  tList, fList, pClass = nbcTrain(train)
  result, tProbList = nbcPredict(test, tList, fList, pClass)

  zeroOneLoss = 0.0
  sqLoss = 0.0

  for row in range(len(test)):
    if test[row][-1] != result[row]:
      zeroOneLoss += 1

    pi = 0.0
    if result[row] == 1:
      pi = tProbList[row]
    else:
      pi = 1 - tProbList[row]
    sqLoss += (1 - (pi * pi))
    
  zeroOneLoss /= len(test)
  sqLoss /= len(test)

  print("ZERO-ONE LOSS=%.4f" % zeroOneLoss)
  print("SQUARED LOSS=%.4f" % sqLoss)
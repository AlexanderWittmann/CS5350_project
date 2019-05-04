import collections
import csv
import random
import sys

def getData():
    with open('spam.csv','r') as testingData:
        csvReader = csv.reader(testingData)
        allData = []
        for row in csvReader:
            allData.append(row)

    return allData
def getNewData():
    with open('SpamWithFeatures.csv','r') as testingData:
        csvReader = csv.reader(testingData)
        allData = []
        temp = [1]
        for row in csvReader:
            row = temp + row
            allData.append(row)

    for i in range(len(allData)):
        for j in range(len(allData[i])):
            allData[i][j] = float(allData[i][j])
    return allData
def mostCommonWords(Data):
    hamString = ""
    spamString = ""
    for line in Data:
        if line[0] == "ham":
            hamString += line[1]
        elif line[0] == "spam":
            spamString += line[1]

    hamWords = {}
    spamWords = {}
    # Words to ignore from the 20 most common words
    # you, a, for, your, and, have, is, in, on
    # These are all words that people would commonly use in a sms message and I chose to ignore them
    for word in hamString.lower().split():
        word = word.replace(".","")
        word = word.replace(",","")
        word = word.replace(":","")
        word = word.replace("!","")
        word = word.replace("you","")
        word = word.replace("a","")
        word = word.replace("for","")
        word = word.replace("your","")
        word = word.replace("and","")
        word = word.replace("have","")
        word = word.replace("is","")
        word = word.replace("in","")
        word = word.replace("on","")
        if word == "":
            continue
        if word not in hamWords:
            hamWords[word] = 1
        else:
            hamWords[word] += 1

    for word in spamString.lower().split():
        word = word.replace(".","")
        word = word.replace(",","")
        word = word.replace(":","")
        word = word.replace("!","")
        word = word.replace("you","")
        word = word.replace("a","")
        word = word.replace("for","")
        word = word.replace("your","")
        word = word.replace("and","")
        word = word.replace("have","")
        word = word.replace("is","")
        word = word.replace("in","")
        word = word.replace("on","")
        if word == "":
            continue
        if word not in spamWords:
            spamWords[word] = 1
        else:
            spamWords[word] += 1
    mostCommonHam = collections.Counter(hamWords).most_common(20)
    mostCommanSpam = collections.Counter(spamWords).most_common(20)
    return (mostCommanSpam,mostCommonHam)

def newDataSet(Data):
    newdata = []
    baseline = [0,0,0,0,0,0,0,0,-1]
    for line in Data:
        temp = [0,0,0,0,0,0,0,0,-1]
        if line[0] == "spam":
            temp[-1] = 1
        if "free" in line[1].lower():
            temp[0] = 1
        elif "txt" in line[1].lower():
            temp[1] = 1
        elif "mobile" in line[1].lower():
            temp[2] = 1
        elif "cliam" in line[1].lower():
            temp[3] = 1
        elif "reply" in line[1].lower():
            temp[4] = 1
        elif "my" in line[1].lower():
            temp[5] = 1
        elif "now" in line[1].lower():
            temp[6] = 1
        elif "but" in line[1].lower():
            temp[7] = 1
        tempData = str(temp[0])
        for i in range(1,9):
            tempData += ", " + str(temp[i])
        tempData += "\n"
        newdata.append(tempData)
    newDataFile = open("SpamWithFeatures.csv","w")
    newDataFile.writelines(newdata)
    newDataFile.close()

def updateW(W,data,R,type):
    newW = []
    if type:
        for i in range(5):
            update = W[i] + R*data[i]
            newW.append(update)
    else:
        for i in range(5):
            update = W[i]-R*data[i]
            newW.append(update)
    return newW

def checkUpdate(W,data):

    result = 0.0
    for i in range(5):
        result += (W[i]*data[i])
    #result = result * data[-1]

    return result

def main():
    Data = getData()
    FeatureData = getNewData()
    # Only need to do this once to get the new dataSet
    #newDataSet(Data)
    (commonS,commonH) = mostCommonWords(Data)
    print("The most common words in normal SMS messages: ",commonH)
    print("The most common words in spam SMS messages: ",commonS)
    fileSpam = open("SpamWords.txt","w")
    fileHam = open("HamWords.txt","w")
    for word in commonH:
        temp = word[0] + ", " + str(word[1]) + "\n"
        fileHam.write(temp)

    for word in commonS:
        temp = word[0] + ", " + str(word[1]) + "\n"
        fileSpam.write(temp)
    fileHam.close()
    fileSpam.close()


    # Now that I have a data set with features I am going to run the Perceptron algorithm

    R = 0.01
    iterations = 10

    # initial W
    W = [0,0,0,0,0,0,0,0,0]
    allW = []
    allW.append((W,1))
    pos = True
    neg = False

    for i in range(iterations):
        # for each epoch
        # shuffle the data
        shuffle = FeatureData
        random.shuffle(shuffle)

        # for each training example (xi,yi) in training data654rt
        # check each yiwtxi<0, update W<-w+ryixi
        for line in shuffle:
            check = checkUpdate(W,line)
            if line[-1] < 0:
                if check >= 0:
                    W = updateW(W,line,R,neg)
                    allW.append((W,1))
                else:
                    # add if it correclty predicts the result
                    temp = allW[-1][1]
                    temp += 1
                    temp1 = allW[-1][0]
                    allW[-1] = (temp1,temp)
            else:
                if check < 0:
                    W = updateW(W,line,R,pos)
                    allW.append((W,1))
                else:
                    # add to C if it correctly predicts the result
                    temp = allW[-1][1]
                    temp += 1
                    temp1 = allW[-1][0]
                    allW[-1] = (temp1,temp)

    # now we will test the training data and see how the linear classifier works
    correct = 0
    for line in FeatureData:
        prediction = 0
        for aW in allW:
            temp = 0
            for i in range(5):
                temp = aW[0][i]*line[i]
            prediction += temp * aW[1]
        if (line[-1]*prediction) >= 0:
            correct += 1


    print("Last predicted W: ",allW[-1])
    print("Error on Training Data: ",correct,"/",len(FeatureData))
    sys.exit()


if __name__=="__main__":
    main()
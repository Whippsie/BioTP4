from ete3 import Tree
import numpy

# Pour lire le fichier contenant les arbres
def readTrees(filename):
    trees = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip("\n")
            if "(" in line and ")" in line:
                t = Tree(line)
                print (t)
                trees.append(t)
    return trees

# Fonction pour calculer la distance RF entre deux arbres
def robinsonFould(tree1, tree2):
    # Initialisation
    distance = 0
    childrenNum = min(len(tree1), len(tree2))
    distanceMax = 2*(childrenNum - 3)

    # On deracine les arbres
    t1 = tree1
    t2 = tree2
    t1.unroot()
    t2.unroot()

    children1 = t1.children
    children2 = t2.children

    # Comparaisons de toutes les bipartitions
    # On assume que de la "racine" il y aura toujours 3 enfants
    # Car les arbres déracinés sont obtenus d'arbres binaires
    # Par contre, pour être prudent, seulement le nombre minimal
    # De branches à partir de la "racine" va dicter combien de
    # Bipartitions nous allons faire
    branchNum = min(len(children1), len(children2))
    for i in range(0, branchNum):
        rightElement1 = children1[i].get_leaves()
        rightElement2 = children2[i].get_leaves()

        leftElementsArray1 = []
        for elem in children1:
            leftElementsArray1 += elem
        del leftElementsArray1[i]
        leftElement1 = []
        for elem in leftElementsArray1:
            leftElement1 = leftElement1 + elem.get_leaves()

        leftElementsArray2 = []
        for elem in children2:
            leftElementsArray2 += elem
        del leftElementsArray2[i]
        leftElement2 = []
        for elem in leftElementsArray2:
            leftElement2 = leftElement2 + elem.get_leaves()

        bipartition1 = Bipartition(rightElement1, leftElement1)
        bipartition2 = Bipartition(rightElement2, leftElement2)
        distance += bipartition1.compare(bipartition2)
    print(distance)
    return distance/distanceMax

class Bipartition:
    def __init__(self, right, left):
        self.right = right
        self.left = left

    def getRight(self):
        return self.right

    def getLeft(self):
        return self.left

    def compare(self, bipartition2):
        result = 0
        if not arrayCompare(self.right, bipartition2.getRight()):
            result += 1
        if not arrayCompare(self.left, bipartition2.getLeft()):
            result += 1
        return result

# Returns True if arrays contain the same elements and have the same length
# Returns False otherwise
def arrayCompare(array1, array2):
    if len(array1) != len(array2):
        return False
    for element in array1:
        if not (element in array2):
            return False
    return True

def readSequences(filename):
    file = []
    nom = ""
    content = ""
    first = True
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip("\n")
            if '>' in line:
                if first:
                    nom = line.strip(">")
                    first = False
                else:
                    file.append(Sequence(nom, content, ""))
                    content = ""
                    nom = line.strip(">")
            else:
                content += line
        file.append(Sequence(nom, content, ""))
    return file


class Sequence:
    def __init__(self, name, oldContent, newContent):
        self.name = name
        self.oldContent = oldContent
        self.newContent = newContent

    def getName(self):
        return self.name

    def getOldContent(self):
        return self.oldContent

    def getNewContent(self):
        return self.newContent

    def setNewContent(self, ct):
        self.newContent += ct


def removeGaps(seqList):
    tag = False

    # Pour chaque position des séquences(on suppose même longueur)
    for i in range(len(seqList[0].getOldContent())):
        # Pour chaque séquence de kla liste
        for j in range(len(seqList)):
            if seqList[j].getOldContent()[i] == "-":
                tag = True
        if tag:
            tag = False
        else:
            for l in range(len(seqList)):
                seqList[l].setNewContent(seqList[l].getOldContent()[i])
    return seqList

def updateNJTree(i,j,n,distanceMatrix,seqList):
    # Create new node to join
    sumOne, sumTwo = findSums(i,j,n,distanceMatrix)
    distIToNewNode = 0.5*(float(distanceMatrix[i][j]))+((1/(2*(n-2)))*(sumOne-sumTwo))
    distJToNewNode = float(distanceMatrix[i][j]) - distIToNewNode

    stringList=[]
    #stringList.append(distanceMatrix[i][0] + "-" + distanceMatrix[0][j])
    for s in seqList:
        if s.getName()!= distanceMatrix[i][0] and s.getName() != distanceMatrix[0][j]:
            stringList.append(s.getName())
        elif s.getName() == distanceMatrix[i][0]:
            stringList.append(distanceMatrix[i][0] + "-" + distanceMatrix[0][j])
    n=len(seqList)-1
    newDistanceMatrix = makeNewDistanceMatrix(n,stringList,distanceMatrix,i)
    print("NEW DISTANCE WITH 0 WHERE NEW COLUMN")
    printMatrix(newDistanceMatrix)

    for k in range(i,n):
        for l in range(j,n):
            if k != l:
                newDistanceMatrix[k][l]=0.5*(float(distanceMatrix[i][l])+float(distanceMatrix[j][l])-float(distanceMatrix[i][j]))
    print("")
    print("NEW DISTANCE WITH ALL VALUES")
    printMatrix(newDistanceMatrix)

def testNumpy(distanceMatrix,i):
    numArray = numpy.array(distanceMatrix)
    numArray = numpy.delete(numArray,i,0)
    numArray = numpy.delete(numArray, i, 1)
    numArray = numpy.delete(numArray,i,0)
    numArray = numpy.delete(numArray, i, 1)
    print(numArray)

def makeNewDistanceMatrix(n, seqStringList, distanceMatrix,i):
    newMatrix = []
    rows = n
    columns = rows
    for row in range(rows + 1):
        rowScore = []
        for column in range(columns + 1):
            if row == 0 and column == 0:
                rowScore.append("~")
            elif row == 0:
                rowScore.append(seqStringList[column - 1])
            elif column == 0:
                rowScore.append(seqStringList[row - 1])
            elif row !=i and column!= i:
                rowScore.append(distanceMatrix[row][column])
            else:
                rowScore.append(0)
        newMatrix.append(rowScore)
    return newMatrix


def calculateNJMatrix(seqList, distanceMatrix):
    smallest = float('inf')
    pos = (0,0)
    njMatrix = makeDistanceMatrix(seqList)
    n = len(seqList)
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i!= j:
                temp = (n-2)* float(distanceMatrix[i][j])
                sumOne,sumTwo = findSums(i,j,n,distanceMatrix)
                njMatrix[i][j] = round(temp-sumOne-sumTwo,2)
                if njMatrix[i][j] < smallest:
                    smallest = njMatrix[i][j]
                    pos = (i,j)
    return(njMatrix, pos)

def findSums(i,j,n,distanceMatrix):
    sumOne = 0
    sumTwo = 0
    for k in range(1, n + 1):
        sumOne += float(distanceMatrix[i][k])
        sumTwo += float(distanceMatrix[j][k])
    return sumOne,sumTwo


def calculateDistanceMatrix(blosumMatrix, seqList):
    distanceMatrix = makeDistanceMatrix(seqList)
    # print(distanceMatrix)
    # Parcourt les séquences en pairage 2 à 2
    for i in range(len(seqList)):
        for j in range(len(seqList)):
            p = 0
            qi = 0
            qj = 0
            if i != j:
                # Pour chacune des 2 séquences, regarde les caractères 1 à 1
                for k in range(len(seqList[i].getNewContent())):
                    p += getDistanceP(blosumMatrix, seqList[i].getNewContent()[k], seqList[j].getNewContent()[k])
                    qi += getDistanceQ(blosumMatrix, seqList[i].getNewContent()[k])
                    qj += getDistanceQ(blosumMatrix, seqList[j].getNewContent()[k])
                distanceMatrix[i + 1][j + 1] = round(getBlosumScore(p, qi, qj), 2)
    return distanceMatrix


def makeDistanceMatrix(seqList):
    distanceMatrix = []
    rows = len(seqList)
    columns = rows
    for row in range(rows + 1):
        rowScore = []
        for column in range(columns + 1):
            if row == 0 and column == 0:
                rowScore.append("~")
            elif row == 0:
                rowScore.append(seqList[column - 1].getName())
            elif column == 0:
                rowScore.append(seqList[row - 1].getName())
            else:
                rowScore.append(0)
        distanceMatrix.append(rowScore)
    return distanceMatrix


def getDistanceP(matrix, char1, char2):
    line = findLine(matrix, char1)
    col = findColumn(matrix, char2)
    return int(matrix[line][col])


def getDistanceQ(matrix, char):
    line = findLine(matrix, char)
    col = findColumn(matrix, char)
    return int(matrix[line][col])


def getBlosumScore(p, qi, qj):
    return (1 - ((p) / max(qi, qj)))


def makeBlosumMatrix():
    listBlosum = readBlosum("BLOSUM62.txt")
    blosumMatrix = []
    rows = len(listBlosum[0])
    columns = rows
    minus = False
    for row in range(rows):
        if row == 25:
            break
        rowScore = []
        for column in range(columns):
            if row == 18 and column == 53:
                rowScore.append(11)
            else:
                temp = listBlosum[row][column]
                if temp != " ":
                    if temp != "-":
                        if not minus:
                            rowScore.append(temp)
                        else:
                            minus = False
                            temp = "-" + temp
                            rowScore.append(temp)
                    else:
                        minus = True
        blosumMatrix.append(rowScore)

    return blosumMatrix


def readBlosum(filename):
    content = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip("\n")
            line = line.strip(" ")
            content.append(line)
    return content


def findColumn(matrix, char):
    col = len(matrix[0])
    for i in range(col):
        if matrix[0][i] == char:
            return i
    return float('inf')


def findLine(matrix, char):
    rows = len(matrix[0])
    for i in range(rows):
        if matrix[i][0] == char:
            return i
    return float('inf')


def printSequences(seqList):
    # for s in seqList:
    # print(s.getOldContent())
    for s in seqList:
        print (s.getNewContent())


def printMatrix(matrix):
    for col in matrix:
        print (col)

def main():
    seqList = readSequences("proteines.fa")

    seqList = removeGaps(seqList)
    # print (len(seqList[0].getNewContent()))
    # printSequences(seqList)

    blosumMatrix = makeBlosumMatrix()
    # printMatrix(blosumMatrix)

    distanceMatrix = calculateDistanceMatrix(blosumMatrix, seqList)
    printMatrix(distanceMatrix)

    njMatrix, posSmallest = calculateNJMatrix(seqList,distanceMatrix)
    print(" =========================================")
    printMatrix(njMatrix)
    print("Smallest is: ",posSmallest[0],posSmallest[1])

    # Test
    print(" ===================TESTING======================")
    testList = [['~','a','b','c','d','e'],['a','0','5','9','9','8'],['b','5','0','10','10','9'],['c','9','10','0','8','7'],['d','9','10','8','0','3'],['e','8','9','7','3','0']]
    #testNumpy(testList,2)
    print("DISTANCE INITIALE")
    printMatrix(testList)
    a = Sequence('a',"","")
    b = Sequence('b', "", "")
    c = Sequence('c', "", "")
    d = Sequence('d', "", "")
    e = Sequence('e', "", "")
    seqList = [a,b,c,d,e]
    njMatrix, posSmallest = calculateNJMatrix(seqList, testList)
    print("")
    print("DISTANCE MODIFIEE")
    printMatrix(njMatrix)

    print("Smallest is: ",posSmallest[0],posSmallest[1])
    print("")
    updateNJTree(posSmallest[0],posSmallest[1],len(seqList),testList,seqList)
    treesFromFile = readTrees("arbres.nw")

    testScore = robinsonFould(treesFromFile[0], treesFromFile[0])
    print (testScore)
    testScore = robinsonFould(treesFromFile[0], treesFromFile[1])
    print (testScore)


if __name__ == "__main__":
    main()

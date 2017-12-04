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

        # Les bipartitions pour la seq0 ne marchent pas (voir debugger ici)
        # De plus, on ne doit pas prendre en compte les bipartitions triviales
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

def updateNJTree(t,i,j,n,distanceMatrix,seqList):
    # Create new node to join
    sumOne, sumTwo = findSums(i,j,n,distanceMatrix)
    # TODO:Create new tree here
    distIToNewNode = 0.5*(float(distanceMatrix[i][j]))+((1/(2*(n-2)))*(sumOne-sumTwo))
    distJToNewNode = float(distanceMatrix[i][j]) - distIToNewNode
    u = t.add_child(name=distanceMatrix[i][0] + "-" + distanceMatrix[0][j])
    m1 = u.add_child(name=distanceMatrix[i][0],dist=distIToNewNode)
    m2 = u.add_child(name=distanceMatrix[0][j],dist=distJToNewNode)

    stringList=[]
    for s in seqList:
        if s.getName()!= distanceMatrix[i][0] and s.getName() != distanceMatrix[0][j]:
            stringList.append(s.getName())
        elif s.getName() == distanceMatrix[i][0]:
            stringList.append(distanceMatrix[i][0] + "-" + distanceMatrix[0][j])

    n=len(seqList)-1
    newDistanceMatrix = makeNewDistanceMatrix(n,stringList,distanceMatrix,i)

    for k in range(1,n+1):
        #On remplit la nouvelle ligne
        if i==k:
            newDistanceMatrix[i][k] = 0
        else:
            #TODO: Check si ok avec a qui change pas..doit durement rajouter un if
            newDistanceMatrix[i][k] = 0.5 * (float(distanceMatrix[i][k+1]) + float(distanceMatrix[j][k+1]) - float(distanceMatrix[i][j]))
            newDistanceMatrix[k][i] = newDistanceMatrix[i][k]

    return (stringList,newDistanceMatrix)

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
                if row<i and column<i:
                    # On ne touche pas à l'indice
                    rowScore.append(distanceMatrix[row][column])
                else:
                    if row<i:
                        rowScore.append(distanceMatrix[row][column + 1])
                    elif column <i:
                        rowScore.append(distanceMatrix[row+1][column])
                    else:# On réduit l'indice de 1
                        rowScore.append(distanceMatrix[row+1][column+1])
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

def rootTree(t):
    print("unrooted tree ", t)
    print("")
    r = t.get_midpoint_outgroup()
    print("mid point technic = " , r)

    max = 0
    longNode = None
    #Get farthest node from every node
    for node in t.traverse():
        farthest, dist = node.get_farthest_node()
        print ("The farthest node from ", node.name, "is ", farthest.name, "with dist = ", dist)
        if dist>max:
            max = dist
            longNode = node
    print ("The farthest node ever is ", longNode.name, "with dist = ", max)

def main():
    # Creates a list of Sequence object with the name and content
    seqList = readSequences("proteines.fa")

    # Updates the newContent property with the oldContent without gap
    seqList = removeGaps(seqList)
    print("New sequences")
    printSequences(seqList)
    print(" =========================================")
    # Parses the BLOSUM62 matrix
    blosumMatrix = makeBlosumMatrix()

    # Calculates the first distance matrix using blosum62 score
    distanceMatrix = calculateDistanceMatrix(blosumMatrix, seqList)
    print("Matrice initiale des distances")
    printMatrix(distanceMatrix)

    print(" =========================================")

    print("Matrice pondérée")
    njMatrix, posSmallest = calculateNJMatrix(seqList,distanceMatrix)

    printMatrix(njMatrix)
    print("Smallest is: ",posSmallest[0],posSmallest[1])

    t = Tree()
    # Cette fonction merge 2 séquences en une nouvelle, modifie la liste des sequences et rajoute le noeud dans l'arbre
    newSeqList, newMatrix = updateNJTree(t, posSmallest[0], posSmallest[1], len(seqList), distanceMatrix, seqList)
    print(" =========================================")
    print(" =========================================")
    print("Matrice des distances après 1 itération")
    printMatrix(newMatrix)

    count = 0
    #TODO: Compter avec le len et les colonnes et non un 2 random
    # Le but ici est de looper et de modifier la matrice jusqu'à ce que seulement 2 noeuds restent dans la liste des sequences
    # Dans ce cas-là, on les merge dans une racine vide (car NJ retourne un non-enraciné)
    while count < 2:
        count +=1
        temp = []
        # newSeqList est une liste de String, on doit donc créer les objets correspondants
        for s in newSeqList:
            temp.append(Sequence(s, "", ""))

        #On recalcule la matrice NJ à partir de la nouvelle matrice des distances
        njMatrix, posSmallest = calculateNJMatrix(temp, newMatrix)
        print(" =========================================")

        print("Matrice pondérée")
        printMatrix(njMatrix)
        print("Smallest is: ", posSmallest[0], posSmallest[1])
        print("")
        # Idem à plus haut
        newSeqList, newMatrix = updateNJTree(t, posSmallest[0], posSmallest[1], len(newSeqList), newMatrix, temp)
        print(" =========================================")
        print(" =========================================")
        print("Matrice des distances après "+ str(count+1) +" itérations")
        printMatrix(newMatrix)


    treesFromFile = readTrees("arbres.nw")

    #testScore = robinsonFould(treesFromFile[0], treesFromFile[0])
    #print (testScore)
    rf= treesFromFile[0].robinson_foulds(treesFromFile[1])
    print("rf",rf[0])
    testScore = robinsonFould(treesFromFile[0], treesFromFile[1])
    print (testScore)

    #TESTING ROOTING
    t1 = treesFromFile[0]
    t1.unroot()
    rootTree(t1)

if __name__ == "__main__":
    main()

    """
        # Test
        print(" ===================TESTING======================")
        testDistanceMatrix = [['~','a','b','c','d','e'],['a',0,5,9,9,8],['b',5,0,10,10,9],['c',9,10,0,8,7],['d',9,10,8,0,3],['e',8,9,7,3,0]]
        #testNumpy(testList,2)
        print("DISTANCE INITIALE")
        printMatrix(testDistanceMatrix)
        a = Sequence('a',"","")
        b = Sequence('b', "", "")
        c = Sequence('c', "", "")
        d = Sequence('d', "", "")
        e = Sequence('e', "", "")
        seqList = [a,b,c,d,e]
        njMatrix, posSmallest = calculateNJMatrix(seqList, testDistanceMatrix)
        print("")
        #print("DISTANCE MODIFIEE")
        #printMatrix(njMatrix)

        print("Smallest is: ",posSmallest[0],posSmallest[1])
        print("")

    """

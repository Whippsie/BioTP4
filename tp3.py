# coding=utf-8
from __future__ import division
from ete3 import Tree
import numpy

# Pour lire le fichier contenant les arbres
def readTrees(filename):
    trees = []
    print("Affichage des arbres du fichier arbres.nw")
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

    bpts1 = bipartitions(tree1)
    bpts2 = bipartitions(tree2)

    for bpt1 in bpts1:
        gotMatch = False

        for bpt2 in bpts2:

            if bpt1.compare(bpt2):
                gotMatch = True

        if not gotMatch:
            distance += 2

    return distance

class Bipartition:
    def __init__(self, right, left):
        self.right = right
        self.left = left

    def getRight(self):
        return self.right

    def getLeft(self):
        return self.left

    def compare(self, bpt2):
        if not ((arraysCompare(self.right, bpt2.getRight()) and arraysCompare(self.left, bpt2.getLeft())) \
                or (arraysCompare(self.left, bpt2.getRight()) and arraysCompare(self.right, bpt2.getLeft()))):
            return False
        return True

    def printBipartition(self):
        print (", ".join(self.left) + " | " + ", ".join(self.right))

def arraysCompare(array1, array2):
    if len(array1) != len(array2):
        return False
    for elem in array1:
        if not elem in array2:
            return False
    return True

def bipartitions(tree):
    result = []
    leafs = tree.get_leaf_names()
    isParentRoot = False

    for node in tree.iter_descendants():
        if not node.is_leaf():
            partition1 = node.get_leaf_names()
            partition2 = set(leafs) - set(partition1)

            if len(partition1) > 1 and len(partition2) > 1:
                if node.up.is_root():
                    if not isParentRoot:
                        isParentRoot = True
                        result.append(Bipartition(partition1, partition2))
                else:
                    result.append(Bipartition(partition1, partition2))

    result = getRidOfDuplicates(result)
    print ("Bipartitons trouvees:")
    for bpt in result:
        bpt.printBipartition()

    return result

def getRidOfDuplicates(bpts):
    result = []
    for bpt in bpts:
        if len(result) == 0:
            result.append(bpt)
        else:
            if not bipartitionArrayContains(result, bpt):
                result.append(bpt)
    return result

def bipartitionArrayContains(array, element):
    for bpt in array:
        if bpt.compare(element):
            return True
    return False

def calculateRFMatrix(trees):
    length = len(trees)
    matrix = []
    for i in range(0, length):
        line = []
        for j in range(0, length):
            if i != j:
                print ("-----------------------------RF-------------------------")
                print ("comparaison entre abres :", (i, j))
                print (trees[i])
                print (trees[j])
                res = robinsonFould(trees[i], trees[j])
                print ("result = ", res)
                print ("-----------------------------RF-------------------------")
                line.append(res)
            else:
                line.append(" ")
        matrix.append(line)
    return matrix

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

def updateNJTree(i,j,n,distanceMatrix,seqList,dictPos,dictTree):
    # Calculate summations in advance
    sumOne, sumTwo = findSums(i,j,n,distanceMatrix)

    #BRANCHES
    # Distance to the first branch
    distIToNewNode = 0.5*(float(distanceMatrix[i][j]))+((1/(2*(n-2)))*(sumOne-sumTwo))
    # Distance to the second branch
    distJToNewNode = float(distanceMatrix[i][j]) - distIToNewNode
    dictTree[distanceMatrix[i][0]].add_features(dist=distIToNewNode)
    dictTree[distanceMatrix[0][j]].add_features(dist=distJToNewNode)
    if n==3:
        for m in range(1,n+1):
            #On ajoute la dernière distance
            if m!=i and m!=j:
                dictTree[distanceMatrix[0][m]].add_features(dist=distanceMatrix[i][m]-distIToNewNode)


    stringList=[]
    njTreeStringArray = []
    for s in seqList:
        if s.getName() != distanceMatrix[i][0] and s.getName() != distanceMatrix[0][j]:
            stringList.append(s.getName())
            njTreeStringArray.append(s.getName())
        elif s.getName() == distanceMatrix[i][0]:
            stringList.append(distanceMatrix[i][0] + "-" + distanceMatrix[0][j])
            njTreeStringArray.append("("+distanceMatrix[i][0] + "," + distanceMatrix[0][j]+")")

    n=len(seqList)-1
    newDistanceMatrix,dictPos,dictTree = makeNewDistanceMatrix(n,stringList,distanceMatrix,i,j,dictPos,dictTree)

    for k in range(1,n+1):
        #On remplit la nouvelle ligne
        if i==k:
            newDistanceMatrix[i][k] = 0
        else:
            if k >= j:
                newDistanceMatrix[i][k] = 0.5 * (float(distanceMatrix[i][k+1]) + float(distanceMatrix[j][k+1]) - float(distanceMatrix[i][j]))
            else:
                newDistanceMatrix[i][k] = 0.5 * (float(distanceMatrix[i][k]) + float(distanceMatrix[j][k]) - float(distanceMatrix[i][j]))
            newDistanceMatrix[k][i] = newDistanceMatrix[i][k]

    return stringList,newDistanceMatrix, njTreeStringArray,dictPos,dictTree

def makeNewDistanceMatrix(n, seqStringList, distanceMatrix,i,j,dictPos,dictTree):
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
                #On spécifie la valeur du noeud dans la nouvelle matrix (oldVal,newVal)
                if seqStringList[column - 1] in dictPos:
                    dictPos[seqStringList[column - 1]] = (dictPos[seqStringList[column - 1]][0],column)
                else:
                    # On doit créer un nouvel entrée pour le merge
                    dictPos[seqStringList[column - 1]] = (column, column)
                    t = Tree()
                    t.add_child(dictTree[distanceMatrix[i][0]])
                    t.add_child(dictTree[distanceMatrix[0][j]])
                    t.add_features(name=seqStringList[column - 1],dist=0)
                    dictTree[seqStringList[column - 1]] = t

                    #On doit inactiver les anciennes valeurs

            elif column == 0:
                rowScore.append(seqStringList[row - 1])
            elif row !=i and column!= i and row!= column:
                rowScore.append(distanceMatrix[dictPos[seqStringList[row - 1]][0]][dictPos[seqStringList[column - 1]][0]])
            else:
                rowScore.append(0)
        newMatrix.append(rowScore)

    for row in range(rows + 1):
        # On met à jour les anciens indices
        dictPos[seqStringList[row - 1]] =  (dictPos[seqStringList[row - 1]][1],dictPos[seqStringList[row - 1]][1])

    return newMatrix,dictPos,dictTree


def calculateNJMatrix(seqList, distanceMatrix,dictPos,dictTree):
    smallest = float('inf')
    pos = (0,0)
    njMatrix,dictPos, dictTree = makeDistanceMatrix(seqList,dictPos,dictTree, False)
    n = len(seqList)
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i!= j:
                temp = (n-2)* float(distanceMatrix[i][j])
                sumOne,sumTwo = findSums(i,j,n,distanceMatrix)
                njMatrix[i][j] = temp-sumOne-sumTwo
                if njMatrix[i][j] < smallest:
                    smallest = njMatrix[i][j]
                    pos = (i,j)
    return(njMatrix, pos,dictPos,dictTree)

def findSums(i,j,n,distanceMatrix):
    sumOne = 0
    sumTwo = 0
    for k in range(1, n + 1):
        sumOne += float(distanceMatrix[i][k])
        sumTwo += float(distanceMatrix[j][k])
    return sumOne,sumTwo


def calculateDistanceMatrix(blosumMatrix, seqList, dictPos, dictTree):
    distanceMatrix,dictPos,dictTree = makeDistanceMatrix(seqList, dictPos, dictTree, True)
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
                distanceMatrix[i + 1][j + 1] = getBlosumScore(p, qi, qj)
    return distanceMatrix,dictPos,dictTree


def makeDistanceMatrix(seqList, dictPos, dictTree, createDic):
    distanceMatrix = []
    rows = len(seqList)
    columns = rows
    for row in range(rows + 1):
        rowScore = []
        for column in range(columns + 1):
            if row == 0 and column == 0:
                rowScore.append("~")
            elif row == 0:
                #Crée les dictionnaires INITIAUX
                rowScore.append(seqList[column - 1].getName())
                if (createDic):
                    dictPos[seqList[column - 1].getName()] = (column,column)
                    t = Tree()
                    t.add_features(name=seqList[column - 1].getName(), active=True)
                    dictTree[seqList[column - 1].getName()] = t
            elif column == 0:
                rowScore.append(seqList[row - 1].getName())
            else:
                rowScore.append(0)
        distanceMatrix.append(rowScore)
    return distanceMatrix, dictPos,dictTree


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
    for s in seqList:
        print (s.getNewContent())


def printMatrix(matrix):
    for col in matrix:
        print (col)

class Path:
    def __init__(self, start, end, length):
        self.start = start
        self.end = end
        self.length = length

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_length(self):
        return self.length

def distances(tree):
    file = open("Distances.txt", "w")
    for leaf in tree.traverse():
        if not leaf.is_root():
            end, dist = leaf.get_farthest_node()
            dist_parent = tree.get_distance(leaf.name, leaf._get_up().name)
            file.write("==========Node distance==========\n")
            file.write("Node: " + leaf.name + "\n")
            file.write("Distance from parent: " + str(dist_parent) + "\n")
            file.write("Farthest node: " + end.name + "\n")
            file.write("Distance from farthest node: " + str(dist) + "\n")
            file.write("\n")
    file.close()


def rootTree(tree):
    print ("On root cet arbre:")
    print (tree)
    leaves = tree.get_leaves()
    ## On initialise le résultat à la racine
    longestPath = Path(tree, tree, 0)
    print ("ce qui m'interesse")
    print (tree.get_distance("PCDHA1_Souris", "OR2J3_Humain"))
    for leaf in tree.iter_leaves():
        print ("Leaf:", leaf)
        print ("Leaf.up:", leaf._get_up())
        start = leaf
        end, dist = start.get_farthest_node()
        pathLength = tree.get_distance(start.name, end.name)
        if pathLength > longestPath.get_length():
            longestPath = Path(start, end, pathLength)
            print ("LONGEST PATH:")
            print (longestPath.get_start().name)
            print (longestPath.get_end().name)
            print (longestPath.get_length())
    rightSubtree, leftSubtree = findMidpoint(longestPath, tree)
    if rightSubtree is leftSubtree:
        return tree.set_outgroup(rightSubtree)
    print (rightSubtree)
    print (leftSubtree)

    temp = Tree()
    temp.add_child(rightSubtree)
    temp.add_child(leftSubtree)
    return temp

def findMidpoint(path, tree):
    #initialisation
    print ("On est dans midpoint")
    traveledLength = 0
    rightRoot = path.get_start()
    leftRoot = path.get_start()
    while traveledLength < (path.get_length()/2):
        leftRoot = rightRoot
        if rightRoot.is_root():
            children = rightRoot.get_children()
            for child in children:
                if not child is leftRoot:
                    rightRoot = child
        else:
            rightRoot = rightRoot._get_up()
        traveledLength += numpy.abs(tree.get_distance(rightRoot, leftRoot))
    print ("traveled length", traveledLength)
    if traveledLength == (path.get_length()/2):
        leftRoot = rightRoot
    print ("On sort de midpoint")
    return rightRoot, leftRoot

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

    dictPos={}
    dictTree={}

    # Calculates the first distance matrix using blosum62 score
    distanceMatrix,dictPos,dictTree = calculateDistanceMatrix(blosumMatrix, seqList, dictPos, dictTree)
    print("Matrice initiale des distances")
    printMatrix(distanceMatrix)

    print(" =========================================")

    print("Matrice pondérée")
    negativeMatrix, posSmallest,dictPos,dictTree = calculateNJMatrix(seqList,distanceMatrix,dictPos,dictTree)
    printMatrix(negativeMatrix)
    print("Smallest is: ",posSmallest[0],posSmallest[1])

    # Cette fonction merge 2 séquences en une nouvelle, modifie la liste des sequences et rajoute le noeud dans l'arbre
    seqListString, distanceMatrix, njTreeStringArray,dictPos,dictTree = updateNJTree(posSmallest[0], posSmallest[1], len(seqList), distanceMatrix, seqList, dictPos,dictTree)
    print(" =========================================")
    print(" =========================================")
    print("Matrice des distances après 1 itération")
    printMatrix(distanceMatrix)

    # Le but ici est de looper et de modifier la matrice jusqu'à ce que seulement 2 noeuds restent dans la liste des sequences
    # Dans ce cas-là, on les merge dans une racine vide (car NJ retourne un non-enraciné)
    while len(seqListString)>2:

        seqList = []
        # newSeqList est une liste de String, on doit donc créer les objets correspondants
        for s in seqListString:
            seqList.append(Sequence(s, "", ""))

        #On recalcule la matrice NJ à partir de la nouvelle matrice des distances
        negativeMatrix, posSmallest,dictPos,dictTree = calculateNJMatrix(seqList, distanceMatrix,dictPos,dictTree)
        print(" =========================================")

        print("Matrice pondérée")
        printMatrix(negativeMatrix)
        print("Smallest is: ", posSmallest[0], posSmallest[1])
        print("")
        # Cette fonction merge 2 séquences en une nouvelle, modifie la liste des sequences et rajoute le noeud dans l'arbre
        seqListString, distanceMatrix, njTreeStringArray,dictPos,dictTree = updateNJTree(posSmallest[0], posSmallest[1], len(seqList), distanceMatrix, seqList,dictPos,dictTree)
        print(" =========================================")
        print(" =========================================")
        print("Matrice des distances après itérations")
        printMatrix(distanceMatrix)

    #Il nous reste un seul match à faire, on aura toujours 3 colonnes
    t = Tree()
    t.add_child(dictTree[distanceMatrix[0][2]])
    t.add_child(dictTree[distanceMatrix[0][1]])
    t.add_features(name='notaroot')
    t.add_features(dist=0)
    dictTree['notaroot'] = t


    t = dictTree['notaroot']
    print(" =========================================")
    print("Arbre NJ obtenu")
    print (t)
    print(" =========================================")

    treesFromFile = readTrees("arbres.nw")
    # Enlever les commentaires pour voir toutes les comparaisons RF des arbres du fichier
    #rfMatrix = calculateRFMatrix(treesFromFile)
    #print ("==============RF MATRIX==============")
    #printMatrix(rfMatrix)
    cpt =0
    for i in treesFromFile:
        print("Comparaison RF entre Arbre NJ et Arbre " , cpt, " du fichier.")
        print(robinsonFould(i, t))
        cpt+=1
    distances(t)

if __name__ == "__main__":
    main()


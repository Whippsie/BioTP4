from ete3 import Tree
t1 = Tree("((PCDHA1_Humain, OR2J3_Humain), ((PCDHA1_Rat, PCDHA1_Souris), PCDHA1_Bonobo));")
t2 = Tree()
print (t2)
children1 = t1.children
branchNum = len(children1)
#for i in range(0, branchNum):
    #innerbranch = [];
    #rightElement1 = children1[i].get_leaves()
    #print(rightElement1)
    #print(rightElement1.get_leaves())
nameList = ['a','b','c']
dictPos={}
dictTree={}

for i in range(0,len(nameList)):
    nameNode = nameList[i]
    dictPos[nameNode] = i
    t = Tree()
    a = t.add_child(name=nameNode)
    a.add_features(active=True)
    dictTree[nameNode] = a
    print(a)

#Exemple de merge
nameNode = 'd'
dictPos[nameNode] = 1
#nameList = ['a','d']
noeud = Tree()
#a = t.add_child(name=nameNode)
noeud.add_child(dictTree[nameList[1]])
noeud.add_child(dictTree[nameList[2]])
noeud.add_features(name=nameNode)
dictTree[nameNode] = noeud
test = dictTree[nameNode]
print(test.get_ascii(show_internal=True))
print(noeud.get_ascii(show_internal=True))

print(dictPos)
print(dictTree)
for node in t1:
    if node.is_root():
        print("hello")
    #if not node.is_leaf():
        #innerbranch.append(node)
        #print (node)

#for leaf in t1:
    #print (leaf.name)

#print(t1.get_tree_root())
import dendropy
from anomolyZone.checkAnomolyBL import isAnomlyTree

def PureKingmanTreeConstructor(amount,pop_size=1,minimum=0,maximum=float("+inf"),anomolyOnly=False):
    """
    Generates trees under the unconstrained Kingman’s coalescent process.

    amount: amount of trees to Create
    pop_size: some parameter of dendropy's pure_kingman_tree function
    minimum: minimum tolerable branch length
    maximum: maximum tolerable branch length

    Output: A set of trees
    """
    TaxonNamespace = dendropy.TaxonNamespace(["A","B","C","D"])
    #Generate trees
    trees = set()
    while len(trees) < amount:
        tree = dendropy.simulate.treesim.pure_kingman_tree(TaxonNamespace,pop_size)
        invalid = False
        if(anomolyOnly and not isAnomlyTree(tree)): #Anomly check
            invalid = True
        if(not invalid and (minimum>0 or maximum < float("+inf"))):  #Check branch length constraints
            for edge in tree.edges():
                if (edge.length < minimum and edge.length != 0) or (edge.length > maximum):
                    invalid = True
                    break
        if not invalid:
            trees.add(tree)
    return trees

def newickToStructure(newickTree):
    intervals = newickTree.coalescence_intervals()[4:]
    intervals.sort()
    (a,b,c) = tuple(intervals)
    # outputCommand += " -ej " + coalescentHeight + " " + population1 + " " + population2;
    # outputCommand += " -en " + coalescentHeight + " " + population2 + " " + relativePopsize;
    relativePopsize = 1.0 #1.0
    return f"-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 -ej {a} 1 2 -en {a} 2 {relativePopsize} -ej {b} 2 3 -en {b} 3 {relativePopsize} -ej {c} 3 4 -en {c} 4 {relativePopsize}"

def generate(amount,anomolyOnly=False):
    """
    Inputs: amount of trees
    Output: a set of alpha tree structures
    """
    return [newickToStructure(tree) for tree in PureKingmanTreeConstructor(amount,anomolyOnly=anomolyOnly)]

# def generateMSCommand(tree,N0 = 100000):
#     speciesName2MSName = dict()
#     msName2SpeciesName = dict()
#     nodeTimePopsizeList = list()
#     processTree(tree, N0, speciesName2MSName, msName2SpeciesName, nodeTimePopsizeList)

#     nsam = len(tree.leaf_nodes())
#     nraps = 1
#     npop = len(tree.leaf_nodes())

#     outputCommand = "ms" + " " + nsam + " " + " " + nreps + " -T"

#     rho = 4 * N0 * recombRate * sequenceLength
#     outputCommand += " -r "+rho+" "+sequenceLength

#     outputCommand += " -I " + " 1"*npop

#     for node in tree.leaf_nodes():
#         popIndex = speciesName2MSName.get(node.label)
#         popSize = 10000 #suppose to be per
#         ratio = popSize / N0
#         outputCommand += " -n" + popIndex + " " + ratio

#     edge2population = dict()

#     for bundle in nodeTimePopsizeList:
#         node = bundle.

# def processTree(tree,N0 = 100000,speciesName2MSName,msName2SpeciesName,nodeTimePopsizeList):
#     nodeIndex = 0
#     leafIndex = 1
#     for node in tree.postorder_node_iter():
#         if node.is_leaf():
#             populationIndex = leafIndex
#             leafIndex += 1
#             nodeName = node.label #node.taxon
#             speciesName2MSName[nodeName] = populationIndex
#             msName2SpeciesName[populationIndex] = nodeName

#     for node in tree.postorder_node_iter():
#         generationHeight = node.distance_from_root() #node.distance_from_tip()
#         coalescentHeight = generationHeight / (4 * N0)
#         popSize = 10000 #suppose to be per
#         relativePopSize = popSize/N0
#         nodeTimePopsizeList.add((node, coalescentHeight, relativePopsize))

# def generateStructure(x,y,z):
#     a = z
#     b = z + y
#     c = z + y + x
#     return f"-I 4 1 1 1 1 -n 2 1.0 -n 3 1.0 -n 1 1.0 -n 4 1.0 -ej {a} 1 2 -en {a} 2 {b} -ej {b} 2 3 -en {b} 3 {b} -ej {c} 3 4 -en {c} 4 {b}"
#
# structures = {
#     #'XinhaoTree': "-I 4 1 1 1 1 -n 2 1.0 -n 3 1.0 -n 1 1.0 -n 4 1.0 -ej {2.5} 1 2 -en 2.5 2 4.0 -ej {4.0} 2 3 -en 4.0 3 4.0 -ej {11.25} 3 4 -en 11.25 4 4.0",
#     'O': generateStructure(2.5,1.5,7.25),
#     'A': generateStructure(1.0,1.0,7.25),
#     'B': generateStructure(.5,.5,7.25),
#     'C': generateStructure(.15,.5,7.25),
#     'D': generateStructure(.15,.15,7.25),
#     'E': generateStructure(.1,.1,7.25)
# }

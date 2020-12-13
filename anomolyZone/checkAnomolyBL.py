import numpy as np

# """
# Checks if input tree (recombination string) is an anomoly tree
# """
# def isAnomolyTree(islandModelString):
#     x,y,z = parseIslandModelString(islandModelString)
#     return isAnomolyBL(x, y, z)
#
#
# """
# Parses Island Model String into its x, y, z branch lengths (in coalescent units)
# """
# def parseIslandModelString():
#     return 1,2,3

def percentAnomaly(newickTrees):
    """
    Input: list of newickTrees - dendropy trees
    Output: percent of newickTrees that are in anomoly zone
    """
    total = len(newickTrees)
    numAnomaly = 0
    for newickTree in newickTrees:
        if isAnomolyTree(newickTree):
            numAnomaly += 1
    return numAnomaly/total * 100

def isAnomolyTree(newickTree):
    """
    Input: newickTree - a dendropy tree
    Output: bool - whether or not the tree is in the anomoly zone
    """
    x, y, z = getNewickBL(newickTree)
    return isAnomolyBL(x, y, z)

def getNewickBL(newickTree):
    """
    Input: newickTree - a dendropy trees
    Ouput: x,y, z branch lengths (in coalescent units) of the tree
    """
    intervals = newickTree.coalescence_intervals()[4:]
    intervals.sort()
    (a,b,c) = tuple(intervals)
    z = a
    y = b - a
    x = c - b

    #valid branch lengths (can't be negative)
    assert x >= 0 and y >= 0 and z >= 0

    return x, y, z


def isAnomolyBL(x, y, z=0):
    """
    Checks whether the input branch lengths are in the anomoly zone
    Input: x, y, z branch lengths in coalescent
    Output: True if in anomoly zone; False otherwise
    """

    #valid branch lengths (can't be negative)
    assert x >= 0 and y >= 0 and z >= 0

    #no other bounds on z branch length

    #x branch length bounds
    if x <= 0 or x > 0.27:
        return False

    #y branch length bounds in terms of x: anomoly zone function approximation
    y_bound = anomolyFunc(x)
    if y_bound: #y_bound not false
        return y <= y_bound
    else:
        return False

def anomolyFunc(x):
    """
    Anomoly zone approximation function
    """
    return np.log(2/3 + (3*np.exp(2*x)-2)/(18*(np.exp(3*x)-np.exp(2*x))))

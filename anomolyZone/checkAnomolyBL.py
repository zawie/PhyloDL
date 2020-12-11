
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


def isAnomolyBL(x, y, z=0):
    """
    Checks whether the input branch lengths are in the anomoly zone
    Input: x, y, z branch lengths in coalescent
    Output: True if in anomoly zone; False otherwise
    """
    #x branch length bounds
    if (x > 0.27 or x <= 0):
        return False

    #y branch length bounds in terms of x
    #y_bound anomoly zone function approximation
    y_bound = -0.8 + (10496 / ((1 + (x / 8.423 * (10 ** -17)) ** 0.33)))
    if (y > y_bound):
        return False

    #no bounds on z branch length
    return True

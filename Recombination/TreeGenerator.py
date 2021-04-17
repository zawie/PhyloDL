import dendropy

def PureKingmanTreeConstructor(amount,pop_size=1):
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
        trees.add(tree)
    return trees

def newickToStructure(newickTree):
    intervals = newickTree.coalescence_intervals()[4:]
    intervals.sort()
    (a,b,c) = tuple(intervals)
    # outputCommand += " -ej " + coalescentHeight + " " + population1 + " " + population2;
    # outputCommand += " -en " + coalescentHeight + " " + population2 + " " + relativePopsize;
    relativePopsize = 1.0 #1.0
    return f"-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 \
            -ej {a} 1 2 -en {a} 2 {relativePopsize} \
            -ej {b} 2 3 -en {b} 3 {relativePopsize} \
            -ej {c} 3 4 -en {c} 4 {relativePopsize}"

def generate(amount,name="Unknown"):
    """
    Inputs: amount of trees
    Output: a set of alpha tree structures
    """
    newicktrees = PureKingmanTreeConstructor(amount)
    iStrings = [newickToStructure(tree) for tree in newicktrees]
    return iStrings
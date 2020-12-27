"""
return f"-I 4 1 1 1 1 -n 1 1.0 -n 2 1.0 -n 3 1.0 -n 4 1.0 \
            -ej {a} 1 2 -en {a} 2 {relativePopsize} \
            -ej {b} 2 3 -en {b} 3 {relativePopsize} \
            -ej {c} 3 4 -en {c} 4 {relativePopsize}"
"""

def generate(amount,symmetricPrecent=0.4):
    """
    Input:  amount: amount of trees to create
            symmetricPrecent: the precentage of trees that will be symmetric
    Output: a set of alpha tree structures
    """
    symCount = int(symmetricPrecent*amount)
    asymmCount = amount - symCount
    trees = set()
    for _ in range(symCount):
        trees.add(createSymmetricTree())
    for _ in range(asymCount):
        trees.add(createAsymmetricTree())
    return trees

def createAsymmetricTree():
    """
    Input: None
    Output: A single island model asymmetric tree "
    """
    pass

def createSymmetricTree():
    """
    Input: None
    Output: A single island model symmetric tree "
    """
    pass
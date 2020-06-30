branch_stuff = set([":",".","0","1","2","3","4","5","6","7","8","9"])

def taxon_name_replace(tree):
    tree = tree.replace("taxon1", "A")
    tree = tree.replace("taxon2", "B")
    tree = tree.replace("taxon3", "C")
    tree = tree.replace("taxon4", "D")
    return tree

def getDeepestPair(tree):
    open_count = 0
    coma_count = 0
    A = ""
    B = ""
    tree = taxon_name_replace(tree)

    for char in tree:
        if open_count == 2:
            if char == ",":
                coma_count +=1
            elif char == ")":
                break
            elif char not in branch_stuff:
                if coma_count == 0:
                    A += char
                elif coma_count == 1:
                    B += char
        elif char == "(":
            open_count += 1
    return set((A,B))

def areSame(tree0,tree1):
    p0 = getDeepestPair(tree0)
    p1 = getDeepestPair(tree1)
    return p0 == p1

def getClass(tree):
    tree = taxon_name_replace(tree)
    pair = getDeepestPair(tree)
    partition = {0: [set(("A","B")),set(("C","D"))],
                 1: [set(("A","C")),set(("B","D"))],
                 2: [set(("A","D")),set(("C","B"))]}
    for label,p in partition.items():
        if pair in p:
            return label
    return None

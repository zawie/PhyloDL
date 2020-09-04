branch_stuff = set([":",".","0","1","2","3","4","5","6","7","8","9","e","-","(",")"])

def getDeepestPair(tree,open_threshold=2):
    open_count = 0
    coma_count = 0
    A = ""
    B = ""
    for char in tree:
        if open_count == open_threshold:
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
    return frozenset((A,B))

def getClass(tree):
    #Double parathesis
    for open_threshold in [2,3]:
        pair = getDeepestPair(tree,open_threshold=open_threshold)
        parition = {frozenset(("A","B")):0,
                    frozenset(("C","D")):0,
                    frozenset(("A","D")):2,
                    frozenset(("B","C")):2,
                    frozenset(("A","C")):1,
                    frozenset(("D",'B')):1}
        return parition[pair]

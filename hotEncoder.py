def encode(sequence):
    """
        Hot encodes input sequnce
        "ATGC" -> [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    """
    code_map = {"A":[1,0,0,0],
                "T":[0,1,0,0],
                "G":[0,0,1,0],
                "C":[0,0,0,1]}
    final = []
    for char in sequence:
        final.append(code_map[char])
    return final

def decode(sequence):
    """
        Hot encodes input sequence
        "ATGC" -> [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    """
    code_map = {(1,0,0,0):"A",
                (0,1,0,0):"T",
                (0,0,1,0):"G",
                (0,0,0,1):"C"}
    final = ""
    for char in sequence:
        final += code_map[tuple(char.tolist())]
    return final

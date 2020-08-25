import torch
toXTensor = lambda sequence: torch.tensor(sequence,dtype=torch.float)
toYTensor = lambda n: torch.tensor(n,dtype=torch.long)

#Alpha conveteres
def alphaToBeta(alphaSeqeunces):
    """
    Transforms a given alpha sequences to a beta tree
    """
    (A,B,C,D) = alphaSeqeunces
    return [A,D,C,B]

def alphaToGamma(alphaSeqeunces):
    """
    Transforms a given alpha sequences to a gamma tree
    """
    (A,B,C,D) = alphaSeqeunces
    return [A,C,B,D]

#Beta converters
def betaToAlpha(betaSequences):
    """
    Transforms a given beta sequences to a alpha tree
    """
    (A,D,C,B) = betaSequences
    return [A,B,C,D]

def betaToGamma(betaSequences):
    """
    Transforms a given beta sequences to a gamma tree
    """
    (A,D,C,B) = betaSequences
    return [A,C,B,D]

#Gamma converters
def gammaToAlpha(gammaSequences):
    """
    Transforms a given gamma sequences to an alpha tree
    """
    (A,C,B,D) = gammaSequences
    return [A,B,C,D]

def gammaToBeta(gammaSequences):
    """
    Transforms a given gamma sequences to a beta tree
    """
    (A,C,B,D)= gammaSequences
    return [A,D,C,B]

#Meta-converters
def transformAlpha(alphaSequences):
    return [alphaSequences,alphaToBeta(alphaSequences),alphaToGamma(alphaSequences)]
def transformBeta(betaSequences):
    return [betaToAlpha(betaSequences),betaSequences,betaToGamma(betaSequences)]
def transformGamma(gammaSequences):
    return [gammaToAlpha(gammaSequences),gammaToBeta(gammaSequences),gammaSequences]

def transformSequences(tensorSequences,label):
    #Tensor -> List
    sequenceSet = tensorSequences.tolist()
    label = label.tolist()
    #Perform switch
    switch = {0:transformAlpha,1:transformBeta,2:transformGamma}
    listOfSequenceSets = switch[label](sequenceSet)
    #List -> tensor
    tensorSequenceSets = list()
    for sequenceSet in listOfSequenceSets:
        tensorSequenceSets.append(toXTensor(sequenceSet))
    #Labels
    labels = [toYTensor(0),toYTensor(1),toYTensor(2)]
    return (tensorSequenceSets,labels)

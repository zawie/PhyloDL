import torch
toXTensor = lambda sequence: torch.tensor(sequence,dtype=torch.float)
toYTensor = lambda n: torch.tensor(n,dtype=torch.long)

#Alpha conveteres
def alphaToGamma(alphaSeqeunces):
    """
    Transforms a given alpha sequences to a beta tree
    """
    (A,B,C,D) = alphaSeqeunces
    return [A,D,C,B]

def alphaToBeta(alphaSeqeunces):
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
    (A,C,B,D) = betaSequences
    return [A,B,C,D]

def betaToGamma(betaSequences):
    """
    Transforms a given beta sequences to a gamma tree
    """
    (A,C,B,D) = betaSequences
    return [A,D,C,B]

#Gamma converters
def gammaToAlpha(gammaSequences):
    """
    Transforms a given gamma sequences to an alpha tree
    """
    (A,D,C,B) = gammaSequences
    return [A,B,C,D]

def gammaToBeta(gammaSequences):
    """
    Transforms a given gamma sequences to a beta tree
    """
    (A,D,C,B)= gammaSequences
    return [A,C,B,D]

#Meta-converters
def transformAlpha(alphaSequences):
    return [alphaSequences,alphaToBeta(alphaSequences),alphaToGamma(alphaSequences)]
def transformBeta(betaSequences):
    return [betaToAlpha(betaSequences),betaSequences,betaToGamma(betaSequences)]
def transformGamma(gammaSequences):
    return [gammaToAlpha(gammaSequences),gammaToBeta(gammaSequences),gammaSequences]

def transformSequences(sequenceSet,label):
    #Tensor -> List
    label = label.tolist()
    sequenceSet = sequenceSet.tolist()
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

def transformData(data,labels):
    X = list()
    Y = list()
    for datapoint in zip(data,labels):
        (sequences,label) = datapoint
        (transX,transY) = transformSequences(sequences,label)
        X += transX
        Y += transY
    return(X,Y)

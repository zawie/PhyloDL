import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import checkAnomolyBL

def anomolyDistTrees(newickTrees, name="Anomoly Zone"):
    """
    Input: list of newickTrees - a dendropy tree
    Output: plt of x, y branch length (in coalescent units) distribution
    """
    #get x, y branch lengths
    anomolyX = []
    anomolyY = []
    for newickTree in newickTrees:
        x, y, z = checkAnomolyBL.getNewickBL(newickTree)
        anomolyX.append(x)
        anomolyY.append(y)

    #plot x,y branch length distribution
    anomolyDist(anomolyX, anomolyY, name)

def anomolyDist(anomolyX, anomolyY, name="Anomoly Zone", anomolyApprox=checkAnomolyBL.anomolyApprox):
    """
    Graphs distribution of anomoly zone branch length
    Input:
    - anomolyX: List of x branch lengths in coalescent units
    - anomolyY: List of y branch lengths in coalescent units
    - anomolyApprox: function of anomoly approximation
    Ouput: Graph of distribution of anomoly zone data (x,y) branch lengths
    """

    #generate more datapoints in anomolyzone
    x = np.linspace(start=0, stop = 0.27, num=1000)
    y = anomolyApprox(x)

    #graph range
    x_range = [0, max(max(anomolyX), max(x))]
    y_range = [0, max(max(anomolyY), max(y))]

    #build figure
    fig = go.Figure()
    fig.update_layout(title=name, title_x=0.25, title_y=0.85,
                    xaxis_title="x Branch Length",yaxis_title="y Branch Length",
                    xaxis_range=x_range,yaxis_range=y_range)

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',fill='tozeroy',
                            name = 'Anomoly Zone Approximation'))
    fig.add_trace(go.Scatter(x=anomolyX, y=anomolyY,
                            mode='markers',name='Anomoly Datapoints'))

    #fig.show()

    #save figure
    fig.write_image("anomolyZone/distPlots/" + name + '.png')


    # print("[PLOTTING]")
    # #anomoly graph details
    # plt.title(name)
    # plt.xlabel("x Branch Length")
    # plt.ylabel("y Branch Length")
    # # plt.xlim(0, 0.27)
    # # plt.ylim(0, 1.1)
    #
    # #anomoly zone function approximation plot
    # x = np.linspace(start=0, stop = 0.27, num=1000)
    # y = anomolyApprox(x)
    # plt.plot(x, y, c="darkblue")
    # plt.fill_between(x,y,color="lightblue")
    #
    # #graph anomoly data points
    # plt.scatter(anomolyX, anomolyY, c='red',marker=".", s=30)
    #
    # #save figure
    # plt.savefig("anomolyZone/distPlots/" + name)
    # plt.close('all')

if __name__ == "__main__":
    anomolyX = [0.26,0.025,0.05,0.1,0.2]
    anomolyY = [0,1,0.6,0.3,0.07]

    #generate more datapoints in anomolyzone
    N = 5000
    x = np.random.rand(N) * 0.27
    y= np.random.rand(N)

    for i in range(len(x)):
        if checkAnomolyBL.isAnomolyBL(x[i], y[i]):
            anomolyX.append(x[i])
            anomolyY.append(y[i])

    anomolyDist(anomolyX, anomolyY)

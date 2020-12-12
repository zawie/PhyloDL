import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import checkAnomolyBL

def anomolyDistTrees(newickTrees, name="Anomaly Zone"):
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

def anomolyDist(anomolyX, anomolyY, name="Anomaly Zone", anomolyApprox=checkAnomolyBL.anomolyApprox):
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
    x_range = [0, max(max(anomolyX), max(x), 0.28)]
    y_range = [0, max(max(anomolyY), max(y),1.1)]

    #markersize
    num_points = len(anomolyX)
    markersize = 2
    if num_points <= 10000:
        markersize=3
    elif num_points <=30000:
        markersize=2
    else:
        markersize =1

    #build figure
    fig = go.Figure()
    fig.update_layout(title=name, title_x=0.4,
                    xaxis_title="x Branch Length",yaxis_title="y Branch Length",
                    xaxis_range=x_range,yaxis_range=y_range)

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',fill='tozeroy',
                            name = 'Anomaly Zone Approximation',
                            line_color='DarkBlue',
                            fillcolor='LightBlue'))
    fig.add_trace(go.Scatter(x=anomolyX, y=anomolyY,
                            mode='markers',name='Anomaly Datapoints',
                            marker_color='DarkRed', marker_size=markersize))

    # fig.show()

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
    # anomolyX = [0.26,0.025,0.05,0.1,0.2]
    # anomolyY = [0,1,0.6,0.3,0.07]
    anomolyX = []
    anomolyY = []

    for N in range(10000, 50001, 10000):
        print(N)
        #generate more datapoints in anomolyzone
        #N = 10000
        x = np.random.rand(N) * 4
        y= np.random.rand(N) * 3

        for x_val, y_val in zip(x,y):
            if checkAnomolyBL.isAnomolyBL(x_val, y_val):
                anomolyX.append(x_val)
                anomolyY.append(y_val)

        anomolyDist(anomolyX, anomolyY)

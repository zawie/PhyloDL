# Recombination

Recombination Data Generation:

  1. Change recombination parameters in Recombination/data_generation/ctrlGenPar.py
    ~ Recombination/data_generation/MSTreeTopologyCommands.txt can help you choose the correct MS Command for the tree topology that you want

  2. Change data generation parameters in Recombination/data_generation/main.py

    i) labels - each tree type must be run seperately right now
    ii) dataset sizes & proportions
    iii) sequence lengths, etc.

  3. Run Recombination/data_generation/main.py

  4. Run Recombination/data_generation/recombinationMerge.py making sure all data is in Recombination/data_generation/recombination_data

  5. If there are no lost datapoints, delete that file from Recombination/data_generation/recombination_data

  6. Move output datasets to Recombination/data_generation/test/data

Recombination Data Network/Testing:

  a) Residual Neural Network

    1. In Recombination/recombination_networks/recombination_ResNet.py, change the following parameters:

      a) graph_title
      b) graph_win
      c) data_test
      d) model_number

    2. Run "visdom" in terminal

    3. Type in visdom url into a browser

    4. Run Recombination/recombination_networks/recombination_ResNet.py

  b) IQTREE ML Test

    1. In IQTREE_ML/ML_labeledData.py, change the parameter data_path to the path of your dev set
    2. Run IQTREE_ML/ML_labeledData.py and the IQTREE ML accuracy will be printed in the terminal

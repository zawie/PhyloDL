import sys
import os
#Get information
folderName = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
baseDirectory = os.getcwd()
#Append to system pathes
sys.path.append(f"{baseDirectory}/{folderName}")
print(sys.path)
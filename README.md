# OpenGBStat

This code executes the stochastic framework for evolving grain staitistics introduced in paper (x). 
The input to the framework is a joint probability distribution F(A,S,t=0) of grain area
and number of sides, and the output is its time evolution F(A,S,t).

The overall structure of the code files is as follow. 

MC_main.py: A main driver that runs the simulation 
MC_NeuralNetworks.py: A library file. It has necessary functions and Pytorch neural net class. 
NetworkParameter/*.pth: A set of neural network parameter trained from the phase field simulation results

To run this code, one should install some open source libraries. 
Required libraries are summarize in Table~\ref{tab:packages}. 
The code can be executed by the following command line. 

python3 MC_main.py

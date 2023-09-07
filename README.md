Steps:
1) Parition the dataset to emulate different datasets of the respective number of organizations needed
2) On the server side, initialize the global model parameters overriding flower framework's random client initialization
3) Select a suitable number of clients to take part in the training
4) Encrypt the grid data using various algorithms
5) Train the local models (i.e data from individual organization respectively)
6) Pass the paramters to the server
7) Aggregate using FedAdam strategy
8) Repeat steps 3 to 7 until the number of training rounds are finished

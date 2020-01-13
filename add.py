import tabpy_client
client = tabpy_client.Client("http://localhost:9004/")

def add(x,y):
    import numpy as np
    return np.add(x,y).tolist()

client.deploy('add',add,'Adds x and y', override=True)
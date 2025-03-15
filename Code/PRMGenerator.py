import Franka
import numpy as np
import random
import pickle
import RobotUtil as rt
import time

random.seed(13)

#Initialize robot object
mybot=Franka.FrankArm()

#Create environment obstacles - # these are blocks in the environment/scene (not part of robot) 
pointsObs=[]
axesObs=[]

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,0,1.0]),[1.3,1.4,0.1])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,-0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1, 0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[-0.5, 0, 0.475]),[0.1,1.2,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

# Central block ahead of the robot
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

prmVertices=[] # list of vertices
prmEdges=[] # adjacency list (undirected graph)
start = time.time()

# TODO: Create PRM - generate collision-free vertices
# TODO: Fill in the following function using prmVertices and prmEdges to store the graph. 
# The code at the end saves the graph into a python pickle file.
def PRMGenerator():
    global prmVertices
    global prmEdges
    global pointsObs
    global axesObs
    
    pointsObs = np.array(pointsObs)
    axesObs = np.array(axesObs)
    
    prob_uniform = 0.5
    prob_gaussian = 0.2
    prob_bridge = 0.3
    
    while len(prmVertices)<1000:
        
       # sample next configuration
        if len(prmVertices) < 2 or random.random() < prob_uniform:
            q_new = mybot.SampleRobotConfig()
        elif random.random() < prob_gaussian:
            q_new = SampleRobotConfigGaussian(prmVertices)
        else:
            q_new = SampleRobotConfigBridge(prmVertices)
        
        q_new = mybot.SampleRobotConfig()
        
        # collision checking
        if mybot.DetectCollision(q_new, pointsObs, axesObs):
            continue
        
        # append vertex to graph
        prmVertices.append(q_new)
        # print("Vertex added: ", q_new)
        # append empty list for edges of this vertex
        prmEdges.append([])
        
        # find nearest neighbors by iterating through all vertices and checking distance
        for i in range(len(prmVertices)):
            if np.linalg.norm(np.array(q_new) - np.array(prmVertices[i])) <= 2:
                # point should be different from the neighbor
                if np.array_equal(q_new, prmVertices[i]):
                    continue
                # neighbor found
                # check for edge collision
                if not mybot.DetectCollisionEdge(q_new, prmVertices[i], pointsObs, axesObs):
                    prmEdges[-1].append(i)
                    prmEdges[i].append(len(prmVertices)-1)
    
    #Save the PRM such that it can be run by PRMQuery.py
    f = open("myPRM.p", 'wb')
    pickle.dump(prmVertices, f)
    pickle.dump(prmEdges, f)
    pickle.dump(pointsObs, f)
    pickle.dump(axesObs, f)
    f.close
    
def SampleRobotConfigGaussian(prmVertices, std_dev=0.1):
    # Select a random node
    node_idx = random.randint(0, len(prmVertices) - 1)
    node = prmVertices[node_idx]
    
    # Perturb the node using Gaussian noise
    noise = np.random.normal(0, std_dev, size=len(node))
    new_sample = node + noise
    
    # Ensure the new sample is within joint limits
    new_sample = np.clip(new_sample, mybot.qmin, mybot.qmax)
    
    return new_sample

def SampleRobotConfigBridge(prmVertices):
    # Select two random nodes
    while True:
        node1_idx = random.randint(0, len(prmVertices) - 1)
        node2_idx = random.randint(0, len(prmVertices) - 1)
        
        if not node1_idx == node2_idx:
                break
    
    node1 = prmVertices[node1_idx]
    node2 = prmVertices[node2_idx]
    
    # Generate intermediate sample between the two nodes 
    diff = (np.array(node2) - np.array(node1)) 
    direction = diff / np.linalg.norm(diff)
    sample = node1 + direction * 0.5 * np.linalg.norm(diff)
        
    
    # Ensure the sample is within joint limits
    sample = np.clip(sample, mybot.qmin, mybot.qmax)
    
    return sample

if __name__ == "__main__":

    # Call the PRM Generator function and generate a graph
    PRMGenerator()

    print("\n", "Vertices: ", len(prmVertices),", Time Taken: ", time.time()-start)
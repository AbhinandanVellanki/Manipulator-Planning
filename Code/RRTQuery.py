from random import sample, seed
from re import A
import time
import pickle
import numpy as np
import RobotUtil as rt
import Franka
import time
import mujoco as mj
from mujoco import viewer

# Seed the random object
seed(10)

# Open the simulator model from the MJCF file
xml_filepath = "../franka_emika_panda/panda_with_hand_torque.xml"

np.random.seed(0)
deg_to_rad = np.pi/180.

#Initialize robot object
mybot = Franka.FrankArm()

# Initialize some variables related to the simulation
joint_counter = 0

# Initializing planner variables as global for access between planner and simulator
plan=[]
interpolated_plan = []
plan_length = len(plan)
inc = 1

# Add obstacle descriptions into pointsObs and axesObs
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

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

# define start and goal
deg_to_rad = np.pi/180.

# set the initial and goal joint configurations
qInit = [-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0]
qGoal = [np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0]

# Initialize some data containers for the RRT planner
rrtVertices=[] # list of vertices
rrtEdges=[] # parent of each vertex

rrtVertices.append(qInit)
rrtEdges.append(0)

thresh=0.1
FoundSolution=False
SolutionInterpolated = False

# Utility function to find the index of the nearset neighbor in an array of neighbors in prevPoints
def FindNearest(prevPoints,newPoint):
    D=np.array([np.linalg.norm(np.array(point)-np.array(newPoint)) for point in prevPoints])
    return D.argmin()

# Utility function for smooth linear interpolation of RRT plan, used by the controller
def naive_interpolation(plan):
    angle_resolution = 0.01
    global interpolated_plan 
    global SolutionInterpolated
    interpolated_plan = np.empty((1,7))
    np_plan = np.array(plan)
    interpolated_plan[0] = np_plan[0]
    
    for i in range(np_plan.shape[0]-1):
        max_joint_val = np.max(np_plan[i+1] - np_plan[i])
        number_of_steps = int(np.ceil(max_joint_val/angle_resolution))
        inc = (np_plan[i+1] - np_plan[i])/number_of_steps

        for j in range(1,number_of_steps+1):
            step = np_plan[i] + j*inc
            interpolated_plan = np.append(interpolated_plan, step.reshape(1,7), axis=0)


    SolutionInterpolated = True
    print("Plan has been interpolated successfully!")

#TODO: - Create RRT to find path to a goal configuration by completing the function below. 
def RRTQuery():
    global FoundSolution
    global plan
    global rrtVertices
    global rrtEdges 
    
    goal_bias = 0.1 # probability of sampling q_goal
    del_q = 0.2 # max distance to move towards q_r
    final_connect_threshold = 0.1 # how close does current vertex have to be to the goal to connect directly
    
    while len(rrtVertices)<3000 and not FoundSolution:
        # RRT algorithm to find a path to the goal configuration
        # Use the global rrtVertices, rrtEdges, plan and FoundSolution variables in your algorithm
        
        # sample q_goal with probability goal_bias
        if np.random.rand() < goal_bias:
            q_r = qGoal
        else:
            q_r = mybot.SampleRobotConfig()
            
        # get nearest vertex to q_r from rrtVertices
        q_near_index = FindNearest(rrtVertices, q_r)
        q_near = rrtVertices[q_near_index]
        
        q_c = q_near
        while np.linalg.norm(np.array(q_c) - np.array(q_r)) != 0:
            # get new vertex q_new by moving from q_near towards q_r by max del_q
            diff = np.array(q_r) - np.array(q_c)
            direction = diff/np.linalg.norm(diff)
            step = min(del_q, np.linalg.norm(diff))
            q_c = q_c + step*direction
            
            # check if edge from q_n to q_c is in collision
            if mybot.DetectCollisionEdge(q_near, q_c, pointsObs, axesObs):
                break
            
            if mybot.DetectCollision(q_c, pointsObs, axesObs):
                break
            
            # no collisions, add vertex to rrtVertices and parent index to rrtEdges
            rrtVertices.append(q_c)
            # for i in range(len(rrtVertices)):
            #     if np.all(rrtVertices[i] == q_near):
            #         rrtEdges.append(i)
            #         break
            rrtEdges.append(q_near_index)
            
            # update q_near
            q_near = q_c
            q_near_index = len(rrtVertices) - 1

            # try to connect to goal
            if np.linalg.norm(np.array(q_c) - np.array(qGoal)) < final_connect_threshold and not mybot.DetectCollisionEdge(q_c, qGoal, pointsObs, axesObs):
                # goal reached
                rrtVertices.append(qGoal)
                # for i in range(len(rrtVertices)):
                #     if np.all(rrtVertices[i] == q_c):
                #         rrtEdges.append(i)
                #         break
                rrtEdges.append(q_near_index)
                FoundSolution = True
                break
             
        
    ### if a solution was found
    if FoundSolution:
        # Extract path
        c=-1 #Assume last added vertex is at goal 
        plan.insert(0, rrtVertices[c])

        while True:
            c=rrtEdges[c]
            plan.insert(0, rrtVertices[c])
            if c==0:
                break

        #Path shortening
        # for i in range(150):
        #     # sample two points, one closer to the start than the other
        #     a = np.random.randint(0,len(plan)-1)
        #     b = np.random.randint(a+1,len(plan))
            
        #     # check if edge from a to b is in collision
        #     if mybot.DetectCollisionEdge(plan[a], plan[b], pointsObs, axesObs):
        #         continue
            
        #     # add a and b to plan and remove points between them
        #     plan = plan[:a+1] + [plan[b]] + plan[b+1:]        
    
        for (i, q) in enumerate(plan):
            print("Plan step: ", i, "and joint: ", q)
    
        plan_length = len(plan)	
        naive_interpolation(plan)
        return

    else:
        print("No solution found")

################################# YOU DO NOT NEED TO EDIT ANYTHING BELOW THIS ##############################
def position_control(model, data):
    global joint_counter
    global inc
    global plan
    global plan_length
    global interpolated_plan

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Check if plan is available, if not go to the home position
    if (FoundSolution==False or SolutionInterpolated==False):
        desired_joint_positions = np.array(qInit)
    
    else:

        # If a plan is available, cycle through poses
        plan_length = interpolated_plan.shape[0]

        if np.linalg.norm(interpolated_plan[joint_counter] - data.qpos[:7]) < 0.01 and joint_counter < plan_length:
            joint_counter+=inc

        desired_joint_positions = interpolated_plan[joint_counter]

        if joint_counter==plan_length-1:
            inc = -1*abs(inc)
            joint_counter-=1
        if joint_counter==0:
            inc = 1*abs(inc)
    

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0,0,0,0,0,0,0])

    # Desired gain on position error (K_p)
    Kp = np.eye(7,7)*300

    # Desired gain on velocity error (K_d)
    Kd = 50

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp@(desired_joint_positions-data.qpos[:7]) + Kd*(desired_joint_velocities-data.qvel[:7])


if __name__ == "__main__":

    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    # Set the position controller callback
    mj.set_mjcb_control(position_control)

    # Compute the RRT solution
    RRTQuery()

    # Launch the simulate viewer
    viewer.launch(model, data)
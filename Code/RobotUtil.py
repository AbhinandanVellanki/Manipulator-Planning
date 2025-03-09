import numpy as np 
import math

def rpyxyz2H(rpy,xyz):
    Ht=[[1,0,0,xyz[0]],
        [0,1,0,xyz[1]],
            [0,0,1,xyz[2]],
            [0,0,0,1]]

    Hx=[[1,0,0,0],
        [0,math.cos(rpy[0]),-math.sin(rpy[0]),0],
            [0,math.sin(rpy[0]),math.cos(rpy[0]),0],
            [0,0,0,1]]

    Hy=[[math.cos(rpy[1]),0,math.sin(rpy[1]),0],
            [0,1,0,0],
            [-math.sin(rpy[1]),0,math.cos(rpy[1]),0],
            [0,0,0,1]]

    Hz=[[math.cos(rpy[2]),-math.sin(rpy[2]),0,0],
            [math.sin(rpy[2]),math.cos(rpy[2]),0,0],
            [0,0,1,0],
            [0,0,0,1]]

    H=np.matmul(np.matmul(np.matmul(Ht,Hz),Hy),Hx)
    return H

def R2axisang(R):
    ang = math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
    Z = np.linalg.norm([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    if Z==0:
        return[1,0,0], 0.
    x = (R[2,1] - R[1,2])/Z
    y = (R[0,2] - R[2,0])/Z
    z = (R[1,0] - R[0,1])/Z 	

    return[x, y, z], ang


def BlockDesc2Points(H, Dim):
    center = H[0:3,3]
    axes = [H[0:3,0],H[0:3,1],H[0:3,2]]	
 
    # find corners of the bounding box 3d using dimemsions and axes
    corners=[center,
            center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
            center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
            center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
            center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
            center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
            center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
            center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
            center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.)
         ]   	
    
    # returns corners of BB and axes
    return corners, axes

def CheckPointOverlap(pointsA, pointsB, axis):	
    """
    Inputs:
        - pointsA: 9x3 array of points of box A
        - pointsB: 9x3 array of points of box B
        - axis: 3x1 array of axis to project points on
    
    Outputs:
        - overlap: boolean indicating if there is overlap
    
    """
    # TODO: Project both set of points on the axis and check for overlap
    
    # project points on axis
    projA = np.dot(pointsA, axis)
    projB = np.dot(pointsB, axis)
    
    # find min and max of the projections
    minA, maxA = np.min(projA), np.max(projA)
    minB, maxB = np.min(projB), np.max(projB)
    
    # check for overlap
    no_overlap = minA >= maxB or minB >= maxA
    
    return not no_overlap

def CheckBoxBoxCollision(pointsA, axesA, pointsB, axesB):
    """
    Inputs: 
        - pointsA: 9x3 array of points of box A
        - axesA: 3x3 array of axes of box A representing rotation matrix or direction vectors of surface normals
        - pointsB: 9x3 array of points of box B
        - axesB: 3x3 array of axes of box B representing rotation matrix or direction vectors of surface normals
    
    Outputs:
        - collision: boolean indicating if there is collision
    """	

    #Sphere check
    if np.linalg.norm(pointsA[0]-pointsB[0])> (np.linalg.norm(pointsA[0]-pointsA[1])+np.linalg.norm(pointsB[0]-pointsB[1])):
        return False

    #SAT cuboid-cuboid collision check. 
    #Hint: Use CheckPointOverlap() function to check for overlap along each axis
    
    #Check if cuboids collide along the surface normal of box A
    for i in range(len(axesA)):
        if not CheckPointOverlap(pointsA, pointsB, axesA[i]):
            return False
    
    #Check if cuboids collide along the surface normal of box B
    for i in range(len(axesB)):
        if not CheckPointOverlap(pointsA, pointsB, axesB[i]):
            return False
      
    # get all edges of box A
    edgesA = np.array([[pointsA[0], pointsA[1]], [pointsA[0], pointsA[2]], [pointsA[0], pointsA[4]], [pointsA[1], pointsA[3]], [pointsA[1], pointsA[5]], [pointsA[2], pointsA[3]], [pointsA[2], pointsA[6]], [pointsA[3], pointsA[7]], [pointsA[4], pointsA[5]], [pointsA[4], pointsA[6]], [pointsA[5], pointsA[7]], [pointsA[6], pointsA[7]]])
    
    # get all edges of box B
    edgesB = np.array([[pointsB[0], pointsB[1]], [pointsB[0], pointsB[2]], [pointsB[0], pointsB[4]], [pointsB[1], pointsB[3]], [pointsB[1], pointsB[5]], [pointsB[2], pointsB[3]], [pointsB[2], pointsB[6]], [pointsB[3], pointsB[7]], [pointsB[4], pointsB[5]], [pointsB[4], pointsB[6]], [pointsB[5], pointsB[7]], [pointsB[6], pointsB[7]]])
    
    # get all edge cross products
    edge_cross_prods = np.zeros((edgesA.shape[0], edgesB.shape[0], 3))
    for i in range(edgesA.shape[0]):
        for j in range(edgesB.shape[0]):
            edge_cross_prods[i,j] = np.cross(edgesA[i,1]-edgesA[i,0], edgesB[j,1]-edgesB[j,0])
            
    # check for edge-edge collisions
    for i in range(edge_cross_prods.shape[0]):
        for j in range(edge_cross_prods.shape[1]):
            if not CheckPointOverlap(pointsA, pointsB, edge_cross_prods[i,j]):
                return False
    
    return True

if __name__ == "__main__":
    # Run Test Cases
    test_origins = np.array( [[0,1,0], [1.5,-1.5,0], [0,0,-1], [3,0,0], [-1,0,-2], [1.8,0.5,1.5], [0,-1.2,0.4], [-0.8,0,-0.5]])
    test_ori = np.array([[0,0,0], [1,0,1.5], [0,0,0], [0,0,0], [.5,0,0.4], [-0.2,0.5,0], [0,0.785,0.785], [0,0,0.2]])
    test_dims = np.array([[0.8,0.8,0.8], [1,3,3], [2,3,1], [3,1,1], [2,0.7,2], [1,3,1], [1,1,1], [1,0.5,0.5]])

    ref_origin = np.array([0,0,0])
    ref_ori = np.array([0,0,0])
    ref_dims = np.array([3,1,2])

    ref_bbox_pts, ref_bbox_axis = BlockDesc2Points(rpyxyz2H(ref_ori, ref_origin), ref_dims)
    for i in range(len(test_origins)):
        Hi = rpyxyz2H(test_ori[i], test_origins[i])
        pts_i, ax_i = BlockDesc2Points(Hi, test_dims[i])
        ans_i = CheckBoxBoxCollision(np.array(ref_bbox_pts), np.array(ref_bbox_axis), np.array(pts_i), np.array(ax_i))
        print("Collision between block", i, "and given test block:\t", ans_i)  

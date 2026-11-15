"""Fixed projection copied from the recorded calibrated reader; no text labels."""
import numpy as np

def projection_matrix():
    eye = np.array([.70, -.65, 1.45])
    forward = np.array([0, .15, .65])-eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, [0,0,1])
    side /= np.linalg.norm(side)
    up = np.cross(side, forward)
    view = np.eye(4)
    view[:3,:3] = [side, up, -forward]
    view[:3,3] = [-side@eye, -up@eye, forward@eye]
    f, near, far = 1/np.tan(np.deg2rad(48)/2), .05, 4
    proj = np.array([[f/(768/640),0,0,0],[0,f,0,0],
              [0,0,(far+near)/(near-far),2*far*near/(near-far)],[0,0,-1,0]])
    return proj@view

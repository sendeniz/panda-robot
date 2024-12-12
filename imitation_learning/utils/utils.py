import numpy as np 
import robotic as ry
import numpy as np
from scipy.interpolate import make_lsq_spline

def waypoints_from_bridge_build(joints):

    #joints = np.load(path)
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    pos = []
    quat = []
    for j in joints:
        C.setJointState(j[-1])
        p = C.getFrame("l_gripper").getPosition()
        q = C.getFrame("l_gripper").getQuaternion()
        pos.append(p)
        quat.append(q)

    pos = np.array(pos)
    quat = np.array(quat)

    return pos, quat
    
# Function to create a B-spline fit
def fit_bspline(window_indices, window_data, degree, num_knots):
    # Generate knots
    num_points = len(window_indices)
    if num_points < num_knots + degree + 1:
        return window_indices, window_data  # Not enough points, return raw data
    knots = np.linspace(window_indices[0], window_indices[-1], num_knots)
    t = np.concatenate(([window_indices[0]] * degree, knots, [window_indices[-1]] * degree))
    
    # Create a B-spline fit using least squares
    spl = make_lsq_spline(window_indices, window_data, t, k=degree)
    return window_indices, spl(window_indices)

def fit_polynomial(window_indices, window_data, poly_degree):
    """
    Fits a polynomial of specified degree to the provided data.

    Parameters:
    - window_indices (numpy array): The x-coordinates (indices) of the data points in the window.
    - window_data (numpy array): The y-coordinates (data values) in the window.
    - poly_degree (int): The degree of the polynomial to fit.

    Returns:
    - poly_fit (numpy array): The y-values of the fitted polynomial over the window indices.
    """
    # Fit a polynomial to the data
    poly_coeff = np.polyfit(window_indices, window_data, poly_degree)
    # Evaluate the polynomial
    return np.polyval(poly_coeff, window_indices)

def calculate_num_frames(data_length, window_size, step_size):
    """
    Calculates the number of frames needed for the animation.

    Parameters:
    - data_length (int): Total length of the data.
    - window_size (int): Size of the sliding window.
    - step_size (int): Step size for sliding the window.

    Returns:
    - num_frames (int): Number of frames for the animation.
    """
    return (data_length - window_size) // step_size + 1


def prepare_window_data(data, frame, window_size, step_size):
    """
    Prepares the indices and data for a given frame in a sliding window animation.

    Parameters:
    - data (numpy array): The input data array.
    - frame (int): The current frame index.
    - window_size (int): Size of the sliding window.
    - step_size (int): Step size for moving the window.

    Returns:
    - window_indices (numpy array): Indices of the current window.
    - window_data (numpy array): Data values in the current window.
    """
    start_idx = frame * step_size
    end_idx = min(start_idx + window_size, len(data))
    window_indices = np.arange(start_idx, end_idx)
    window_data = data[start_idx:end_idx]
    return window_indices, window_data

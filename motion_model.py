# velocity motion model
# sampling algorithm 
# in preparation for particle filter implementation
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

# Read the .dat file
df = pd.read_csv("ds1/ds1_Control.dat",sep=r'\s+',comment="#",header=None,names=["time", "forward velocity", "angular velocity"])
df_groundtruth = pd.read_csv("ds1/ds1_Groundtruth.dat",sep=r'\s+',comment="#",header=None,names=["time", "x", "y", "orientation"])

# print(df.head())

def sample(sigma_squared):
    # algo for sampling from approx normal distribution w/
    # zero mean and variance sigma^2

    summation = 0
    sigma = math.sqrt(sigma_squared) 

    for i in range(12):
        summation += random.uniform(-sigma, sigma)

    return 0.5 * summation


def motion_model(u, state_prev,alphas, delta_t):

    # delta_t = 0.1 #timestep

    v, w = u # break down control to linear and rotational vel
    x_prev, y_prev, theta_prev = state_prev
    
    #parameters of the motion noise
    alpha1,alpha2,alpha3,alpha4,alpha5,alpha6= alphas 

    # u = (v w)^T
    v_hat = v + sample(alpha1 * (v ** 2) + alpha2 * (w ** 2))
    w_hat = w + sample(alpha3 * (v ** 2) + alpha4 * (w ** 2))
    gamma_hat = sample(alpha5 * (v ** 2) + alpha6 * (w ** 2))

    # x_prime = x_prev - (v_hat / w_hat) * math.sin(theta_prev) + (v_hat / w_hat) * math.sin(theta_prev + w_hat * delta_t)
    # y_prime = y_prev + (v_hat / w_hat) * math.cos(theta_prev) - (v_hat / w_hat) * math.cos(theta_prev + w_hat * delta_t)
    # theta_prime = theta_prev + w_hat * delta_t + gamma_hat * delta_t

    if abs(w_hat) < 1e-6: #threshold near 0, to avoid dividing by 0 in calculations when w_hat is 0
        x_prime = x_prev + v_hat * delta_t * math.cos(theta_prev) 
        y_prime = y_prev + v_hat * delta_t * math.sin(theta_prev)
        theta_prime = theta_prev + gamma_hat * delta_t
    else :
        x_prime = x_prev - (v_hat / w_hat) * math.sin(theta_prev) + (v_hat / w_hat) * math.sin(theta_prev + w_hat * delta_t)
        y_prime = y_prev + (v_hat / w_hat) * math.cos(theta_prev) - (v_hat / w_hat) * math.cos(theta_prev + w_hat * delta_t)
        theta_prime = theta_prev + w_hat * delta_t + gamma_hat * delta_t

    return (x_prime, y_prime, theta_prime)


if __name__ == '__main__':
    # alpha_params = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    alpha_params = (0,0,0,0,0,0) # perfect path (no noise)
    initial_pose = df_groundtruth.iloc[0] # get initial pose from groundtruth file
    initial_pose = (initial_pose['x'], initial_pose['y'], initial_pose['orientation'])

    # sequence of commands (v, w, t)
    commands = [
        (0.5, 0.0, 1.0),
        (0.0, -1.0 / (2.0 * math.pi), 1.0),
        (0.5, 0.0, 1.0),
        (0.0, 1.0 / (2.0 * math.pi), 1.0),
        (0.5, 0.0, 1.0)
    ]

    # Initialize with the starting pose (known from ground truth file)
    path_history = [initial_pose]
    current_pose = initial_pose

    for v, w, t in commands:
        current_pose = motion_model((v, w), current_pose, alpha_params, t)
        path_history.append(current_pose)
        
    # print("Final Pose:", current_pose)
    # print("Path History:", path_history)

    # Extract x and y coordinates from the path history
    x_coords = [p[0] for p in path_history]
    y_coords = [p[1] for p in path_history]

    plt.figure()
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', label='Robot Path')
    
    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    plt.title('Simulated Robot Path from Motion Model')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensures the scaling of x and y axes are the same
    plt.show()





    # Q3:

    path_history2 = [initial_pose]
    current_pose = initial_pose


    for i in range(len(df) -1):
        curr_t = df.iloc[i]["time"]
        next_t = df.iloc[i+1]["time"]
        dt = next_t-curr_t # calc the timestep

        v = df.iloc[i]["forward velocity"]
        w = df.iloc[i]["angular velocity"]

        current_pose = motion_model((v, w), current_pose, alpha_params, dt)
        path_history2.append(current_pose)

    # print("Final Pose:", current_pose)
    # print("Path History:", path_history2)

    # Extract x and y coordinates from the path history
    x_coords = [p[0] for p in path_history2]
    y_coords = [p[1] for p in path_history2]

    plt.figure()
    plt.plot(x_coords, y_coords, color ='green',linestyle='-', linewidth = 2, label='Robot Path')
        
    # Plot the ground truth path in blue
    plt.plot(df_groundtruth['x'], df_groundtruth['y'], color='blue', linestyle='-', linewidth=2, label='Ground Truth Path')
    

    # Mark start and end points
    # plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    # plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    plt.title('Simulated vs. Ground Truth Robot Path from Motion Model')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensures the scaling of x and y axes are the same
    plt.show()




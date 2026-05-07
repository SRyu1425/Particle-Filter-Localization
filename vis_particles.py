# velocity motion model
# sampling algorithm 
# in preparation for particle filter implementation
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read the .dat file
# delim_whitespace deprecated and replaced by sep
# \s+ means 1+ occurrences of whitespace is delimiter
df_barcodes = pd.read_csv("ds1/ds1_Barcodes.dat",sep=r'\s+',comment="#",header=None,names=["subject", "barcode"])
df_measurements = pd.read_csv("ds1/ds1_Measurement.dat",sep=r'\s+',comment="#",header=None,names=["time", "barcode", "range", "bearing"])
df_landmarks = pd.read_csv("ds1/ds1_Landmark_Groundtruth.dat",sep=r'\s+',comment="#",header=None,names=["subject", "x", "y","x_std_dev","y_std_dev"])

df_control = pd.read_csv("ds1/ds1_Control.dat",sep=r'\s+',comment="#",header=None,names=["time", "forward velocity", "angular velocity"])
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

def motion_model(u, state_prev, delta_t=0.1, alphas=(0.01,0.01,0.01,0.01,0.01,0.01)):

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

    theta_prime = normalize_angle(theta_prime)

    return (x_prime, y_prime, theta_prime)


def prob(a,b2):
    # computes probability of its arg a under a zero centered normal
    # distribution w/ variance b^2 
    return math.exp(-(a**2)/(2*b2)) / (math.sqrt(2 * math.pi * b2))


def create_lookup_tables(df_barcodes, df_landmarks):
    """Create fast lookup dictionaries"""
    barcode_to_subject = dict(zip(df_barcodes['barcode'], df_barcodes['subject']))
    subject_to_landmark = {}
    for _, row in df_landmarks.iterrows():
        subject_to_landmark[row['subject']] = (row['x'], row['y'])
    return barcode_to_subject, subject_to_landmark



# input current robot pose
# input measurement data from df 
def landmark_model_known_correspondence(f_t,x_t, barcode_to_subject, subject_to_landmark):
    # f at time t : observed feature f = (r, phi, s). Where you measured the landmark to be.
    # c at time t : true identity of the feature c ... subject #
    # x at time t : robot pose x = (x y Θ)
    # map m

    r, phi, s = f_t  # the sensor measurement you got
    # rhat is the measurement you should get
    x, y, theta = x_t
    sigma_r, sigma_phi = 0.55,0.55


    # df.loc[rows,cols] is label based lookup
    # rows = mask -> keeps only rows where mask condition is true
    # columns = "some col name" -> returns values from that column
    # .item() extract single value (works b/c we are only expecting 1 element)

    # print("rando:", s)
    # print("real:")
    # input(df_barcodes["barcode"] == s)
    # input(df_barcodes.loc[df_barcodes["barcode"] == s, "subject"].head()) # pd returns series even for 1 elem, so use iloc
    # subject = df_barcodes.loc[df_barcodes["barcode"] == s, "subject"].iloc[0]

    subject = barcode_to_subject[s]

    # # input(df_landmarks.loc[df_landmarks["subject"] == subject, "x"])
    # landmark_x = df_landmarks.loc[df_landmarks["subject"] == subject, "x"].iloc[0]
    # landmark_y = df_landmarks.loc[df_landmarks["subject"] == subject, "y"].iloc[0]

    landmark_x, landmark_y = subject_to_landmark[subject]


    # error between what you measured the dist to landmark to be vs. what it actually is
    r_hat = math.sqrt((landmark_x - x)**2 + (landmark_y - y)**2) 
    phi_hat = math.atan2(landmark_y - y, landmark_x - x) - theta   
    phi_error = normalize_angle(phi - phi_hat)
    q = prob(r - r_hat, sigma_r **2 ) * prob(phi_error, sigma_phi ** 2)
    # probability term for signature error is not included, as it will always be max probability (0 error) 
    # since we know the barcodes of landmarks w/ certainty

    return q

def particle_filter(X_prev, u, z, dt, alpha_params):
    # z may be multiple measurements taken at a single time (multiple landmarks detected)
    X = [] # final return list of particles. Particle will have (x,y,theta). List of triples. 
    X_temp = [] # list of (particle, weight). Before resampling step

    for i in range(len(X_prev)):
        # for each particle 
        x = motion_model(u, X_prev[i], dt, alpha_params) # sample a particle using state transition motion model

        # b/c we can detect multiple landmarks we have to get product of each measurement probability (assumed independence)
        total_weight = 1.0  # Initialize weight for multiplication

        # Loop through each individual measurement for this timestep
        for measurement in z:
            # Calculate probability of observing this single measurement given the particle's state
            prob_one_measurement = landmark_model_known_correspondence(measurement, x)
            
            # Multiply it into the particle's total weight
            total_weight *= prob_one_measurement
 
        X_temp.append((x,total_weight)) #save particle and associated weight in temporary list of states X
        
    # resampling step 

    # Extract particles and weights from X_temp
    particles = [item[0] for item in X_temp]
    weights = [item[1] for item in X_temp]
    num_particles = len(X_temp)

    # Normalize the weights so they sum to 1 and form proper prob distribution
    total_weight_sum = sum(weights)
    weights = [w / total_weight_sum for w in weights]

    # Resample.  draw same num of new particles from the particles list,
    # where the probability of drawing each pose is given by its normalized weight. with replacement.
    indices = np.random.choice(
        np.arange(num_particles), # diff particles available to be chosen (index of particles from particle list)
        size=num_particles, # num of particles to be chosen
        replace=True, # with replacement
        p=weights # associated weights / prob of particle being chosen
    )
    
    X = [particles[i] for i in indices]  # generate list of final states posterior using indices of particles

    return X

def no_filter():
    # running control updates w/o filter
    #### Q2 #######
    # alpha_params = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    alpha_params = (0,0,0,0,0,0) # perfect path (no noise)
    initial_pose = df_groundtruth.iloc[0] # get initial pose from groundtruth file
    initial_pose = (initial_pose['x'], initial_pose['y'], initial_pose['orientation'])


    path_history2 = [initial_pose]
    current_pose = initial_pose


    for i in range(len(df_control) -1):
        curr_t = df_control.iloc[i]["time"]
        next_t = df_control.iloc[i+1]["time"]
        dt = next_t-curr_t # calc the timestep

        v = df_control.iloc[i]["forward velocity"]
        w = df_control.iloc[i]["angular velocity"]

        current_pose = motion_model((v, w), current_pose,dt, alpha_params)
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

def circular_mean(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def initialize_particles(initial_pose, num_particles):
    """Initialize particles with proper diversity"""
    particles = []
    for _ in range(num_particles):
        # Add initial uncertainty
        x = initial_pose[0] + np.random.normal(0, 0.5)  # 10cm std dev
        y = initial_pose[1] + np.random.normal(0, 0.5)  # 10cm std dev
        theta = normalize_angle(initial_pose[2] + np.random.normal(0, 0.1))  # ~3 deg std dev
        particles.append((x, y, theta))
    return particles

if __name__ == '__main__':
    no_filter()
    barcode_to_subject, subject_to_landmark = create_lookup_tables(df_barcodes, df_landmarks)


    # Due to the staggered time steps in the measurement.dat and control.dat, best way to process is combine
    # add a 'type' column to each dataframe to identify the event type
    df_control['type'] = 'control'
    df_measurements['type'] = 'measurement'

    # combine the two dataframes into a single timeline for processing
    all_events = pd.concat([
        df_control[['time', 'forward velocity', 'angular velocity', 'type']],
        df_measurements[['time', 'barcode', 'range', 'bearing', 'type']]
    ])

    ignored_barcodes = {5, 14, 41, 32, 23} # set of barcodes for other 5 moving robots. these are not landmarks and should be ignored.

    # Sort to create the final chronological event timeline. ignore index relabels all rows in new df
    # edge case, if there are measurement rows and a control row at the same time, make sure they are ordered 
    # together so we do not accidentally skip a control 
    all_events = all_events.sort_values(by=['time','type'], ignore_index=True)

    # display first few events
    # print(all_events.head(10))

    # initialization
    num_particles = 750
    alpha_params = (.85,.85,.85,.85,.85,.85)
    # alpha_params = (5,5,5,5,5,5)
    # alpha_params = (0.1,0.1,0.1,0.1,0.1,0.1) # perfect path (no noise)

    initial_pose_df = df_groundtruth.iloc[0] # known init pose assumption
    initial_pose = (initial_pose_df['x'], initial_pose_df['y'], initial_pose_df['orientation'])
    # particles = initialize_particles(initial_pose, num_particles)
    particles = [initial_pose] * num_particles
    mean_pose_history = [initial_pose] # init mean pose tracking for plotting later
    
    # first time from the first event in our sorted list (will be a control)
    last_update_time = all_events.iloc[0]['time']
    # Assume robot is stationary at the beginning
    last_known_control = (0.0, 0.0) 


    # Setup for visualization - only update every N steps for performance
    max_events_to_process = 5000
    plot_interval = 10  # Plot every 10 events
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))
    # ax:  the actual plotting area inside the figure where you draw your data—the canvas itself.
    # All of the subsequent commands, like ax.cla(), ax.scatter(), and ax.plot(), are methods that tell this specific canvas what to draw
    plot_counter = 0 # only plot every plot_interval using modulus

    ax.set_autoscale_on(False)  # Prevents matplotlib from changing limits

    # Determine the full bounds of the map from the ground truth data
    margin = 2.0  # Add 2 meters of padding around the edge
    x_min = df_groundtruth['x'].min() - margin
    x_max = df_groundtruth['x'].max() + margin
    y_min = df_groundtruth['y'].min() - margin
    y_max = df_groundtruth['y'].max() + margin
    

    i = 0
    while i < len(all_events) and i < max_events_to_process:
        current_event = all_events.iloc[i]
        current_time = current_event['time']
        
        dt = current_time - last_update_time
        
        # always do a prediction step based on elapsed time
        if dt > 0:
            # apply last known control for the duration dt to all particles using motion model sample function
            particles = [motion_model(last_known_control, p, dt, alpha_params) for p in particles]

        event_type = current_event['type'] # get event type (control or measurement)

        if event_type == 'control':
            v = current_event['forward velocity']
            w = current_event['angular velocity']
            last_known_control = (v, w) # update curr control command for the next event
            i += 1 # proceed to next
        elif event_type == 'measurement':
            # collect all measurements at this exact timestamp (may be multiple landmarks spotted)
            z_t = []
            start_idx = i
            # Group all measurement rows that share the same timestamp
            while (i < len(all_events) and all_events.iloc[i]['type'] == 'measurement' and all_events.iloc[i]['time'] == current_time):
                row = all_events.iloc[i]
                z_t.append((row['range'], row['bearing'], row['barcode']))
                i += 1
            
            # Now perform the measurement update (weighting and resampling)            
            X_temp = [] # list of (particle, weight). Before resampling step
            for p in particles: # for each particle
                # b/c we can detect multiple landmarks we have to get product of each measurement probability (assumed independence) 
                total_weight = 1.0 # Initialize weight for multiplication
                for measurement in z_t:
                    barcode = measurement[2]  # Barcode is the 3rd element of the tuple (range, bearing, barcode)
                    if barcode in ignored_barcodes:
                        continue
                    total_weight *= landmark_model_known_correspondence(measurement, p, barcode_to_subject, subject_to_landmark)
                X_temp.append((p, total_weight))
            
            # Resampling 
            poses = [item[0] for item in X_temp]
            weights = [item[1] for item in X_temp]
            num_particles = len(X_temp)

            # Normalize the weights so they sum to 1 and form proper prob distribution
            total_weight_sum = sum(weights)
            if total_weight_sum < 1e-9: # Safety check for zero weights
                    weights = [1.0 / num_particles] * num_particles
            else:
                    weights = [w / total_weight_sum for w in weights]
            # Resample.  draw same num of new particles from the particles list,
            # where the probability of drawing each pose is given by its normalized weight. with replacement.
            indices = np.random.choice(
                np.arange(num_particles), # diff particles available to be chosen (index of particles from particle list)
                size=num_particles, # num of particles to be chosen
                replace=True, # with replacement
                p=weights # associated weights / prob of particle being chosen
            )
            
            particles = [poses[i] for i in indices]  # generate list of final states posterior using indices of particles

        # Update the time for the next iteration
        last_update_time = current_time
         
        # save mean pose for plotting
        mean_x = np.mean([p[0] for p in particles])
        mean_y = np.mean([p[1] for p in particles])
        mean_theta = circular_mean([p[2] for p in particles])
        # mean_theta = np.mean([p[2] for p in particles])
        mean_pose_history.append((mean_x, mean_y, mean_theta))

        ## Used to plot the robot live. Tracks particles, ground truth, and actual robot path
        if plot_counter % plot_interval == 0: # if within the interval chosen 
            ax.cla() # clear axes. erase frame and replace with new one for animation 
            # set fixed axes so it doesn't auto scale
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            


            # Plot particles
            px = [p[0] for p in particles]
            py = [p[1] for p in particles]
            ax.scatter(px, py, color='orange', s=10, alpha=0.3, label='Particles') # set size and transparency
            # Plot ground truth up to current time (use proper time filtering)
            current_gt_mask = df_groundtruth['time'] <= current_time
            if current_gt_mask.any():  # Check if we have any ground truth data
                gt_subset = df_groundtruth[current_gt_mask]
                ax.plot(gt_subset['x'], gt_subset['y'], 'b-', linewidth=2, label='Ground Truth')
                
                # Plot current ground truth position
                latest_gt = gt_subset.iloc[-1]
                ax.plot(latest_gt['x'], latest_gt['y'], 'bo', markersize=8, label='GT Current')
            
            # Plot estimated path
            if len(mean_pose_history) > 1:
                hist_x = [p[0] for p in mean_pose_history]
                hist_y = [p[1] for p in mean_pose_history]
                ax.plot(hist_x, hist_y, 'g-', linewidth=2, label='Estimated Path')
                ax.plot(hist_x[-1], hist_y[-1], 'go', markersize=8, label='Est Current')

            # Plot landmarks
            ax.scatter(df_landmarks['x'], df_landmarks['y'], color='k', marker='*', s=150, label='Landmarks')

            ax.set_title(f"Time: {current_time:.2f}s | Event: {i}")
            ax.set_xlabel("X position (m)")
            ax.set_ylabel("Y position (m)")
            ax.legend(loc='upper right') # fix location
            ax.grid(True) # add grid to plot
            
            plt.pause(0.1)  # Slightly longer pause for better visualization
        
        plot_counter += 1


    plt.ioff()

    # Plotting final visualization of total trajectory of ground truth and actual robot
  
    plt.figure(figsize=(12, 8))
    hist_x = [p[0] for p in mean_pose_history]
    hist_y = [p[1] for p in mean_pose_history]
    
    plt.plot(hist_x, hist_y, 'g-', linewidth=2, label='Particle Filter Estimate')
    
    # Plot complete ground truth for comparison
    final_time = all_events.iloc[min(len(all_events)-1, max_events_to_process-1)]['time']
    gt_comparison = df_groundtruth[df_groundtruth['time'] <= final_time]
    plt.plot(gt_comparison['x'], gt_comparison['y'], 'b-', linewidth=2, label='Ground Truth')
    
    plt.scatter(df_landmarks['x'], df_landmarks['y'], color='k', marker='*', s=100, alpha=0.7, label='Landmarks')
    plt.plot(hist_x[0], hist_y[0], 'go', markersize=10, label='Start')
    plt.plot(hist_x[-1], hist_y[-1], 'ro', markersize=10, label='Particle Filter End')
    plt.plot(gt_comparison.iloc[-1]['x'], gt_comparison.iloc[-1]['y'], 'bo', markersize=10, label='Ground Truth End')
    
    plt.title('Final Particle Filter Results')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


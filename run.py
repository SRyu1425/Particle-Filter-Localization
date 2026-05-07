import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read in the .dat files
df_barcodes = pd.read_csv("ds1/ds1_Barcodes.dat",sep=r'\s+',comment="#",header=None,names=["subject", "barcode"])
df_measurements = pd.read_csv("ds1/ds1_Measurement.dat",sep=r'\s+',comment="#",header=None,names=["time", "barcode", "range", "bearing"])
df_landmarks = pd.read_csv("ds1/ds1_Landmark_Groundtruth.dat",sep=r'\s+',comment="#",header=None,names=["subject", "x", "y","x_std_dev","y_std_dev"])
df_control = pd.read_csv("ds1/ds1_Control.dat",sep=r'\s+',comment="#",header=None,names=["time", "forward velocity", "angular velocity"])
df_groundtruth = pd.read_csv("ds1/ds1_Groundtruth.dat",sep=r'\s+',comment="#",header=None,names=["time", "x", "y", "orientation"])

def circular_mean(angles):
    # Calculate the mean of a list of angles correctly
    # Doing regular mean of thetas don't work for angles near 0/2pi boundary
    # mean of [0.1, 6.2] should be ~0.15, not 3.15
    y_sum = np.sum(np.sin(angles)) # take sin of angle list to convert to xy val
    x_sum = np.sum(np.cos(angles))
    return np.arctan2(y_sum, x_sum) # take tan to find angle

def normalize_angle(angle):
    # Normalize angle to [-pi, pi] range so that
    # makes sure 6.28 radians becomes ~0 radians (same direction)
    # for proper angle error calculations in measurement model 
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2  * math.pi
    return angle

def sample(sigma_squared):
    # algo for sampling from approx normal distribution w/ zero mean and variance sigma^2
    summation = 0
    sigma = math.sqrt(sigma_squared) 
    for i in range(12):
        summation += random.uniform(-sigma, sigma)
    return 0.5 * summation

def motion_model(u, state_prev, delta_t=0.1, alphas=(0.01,0.01,0.01,0.01,0.01,0.01)):
    v, w = u # break down control to linear and rotational vel
    x_prev, y_prev, theta_prev = state_prev
    #parameters of the motion noise
    alpha1,alpha2,alpha3,alpha4,alpha5,alpha6= alphas 

    # u = (v w)^T
    v_hat = v + sample(alpha1 * (v ** 2) + alpha2 * (w ** 2))
    w_hat = w + sample(alpha3 * (v ** 2) + alpha4 * (w ** 2))
    gamma_hat = sample(alpha5 * (v ** 2) + alpha6 * (w ** 2))

    if abs(w_hat) < 1e-6: #threshold near 0, to avoid dividing by 0 in calculations when w_hat is 0
        x_prime = x_prev + v_hat * delta_t * math.cos(theta_prev) 
        y_prime = y_prev + v_hat * delta_t * math.sin(theta_prev)
        theta_prime = theta_prev + gamma_hat * delta_t
    else :
        x_prime = x_prev - (v_hat / w_hat) * math.sin(theta_prev) + (v_hat / w_hat) * math.sin(theta_prev + w_hat * delta_t)
        y_prime = y_prev + (v_hat / w_hat) * math.cos(theta_prev) - (v_hat / w_hat) * math.cos(theta_prev + w_hat * delta_t)
        theta_prime = theta_prev + w_hat * delta_t + gamma_hat * delta_t

    theta_prime = normalize_angle(theta_prime) # normalize angle

    return (x_prime, y_prime, theta_prime)

def prob(a,b2):
    # computes probability of its arg a under a zero centered normal distribution w/ variance b^2 
    return math.exp(-(a**2)/(2*b2)) / (math.sqrt(2 * math.pi * b2))

def create_lookup_tables(df_barcodes, df_landmarks):
    # use lookup dicts instead of parsing df (to speed up program significantly)
    barcode_to_subject = dict(zip(df_barcodes['barcode'], df_barcodes['subject'])) # maps barcode to subject #
    subject_to_landmark = {} # maps subject to landmark location
    for _, row in df_landmarks.iterrows(): 
        subject_to_landmark[row['subject']] = (row['x'], row['y'])
    return barcode_to_subject, subject_to_landmark

# input current robot pose
# input measurement data from df 
def landmark_model_known_correspondence(f_t,x_t, barcode_to_subject, subject_to_landmark):
    # Measurement model
    # inputs:
    # f at time t : observed feature f = (r, phi, s). Where you measured the landmark to be.
    # c at time t : true identity of the feature c ... subject (alr have given)
    # x at time t : robot pose x = (x y theta)
    # mappings created above

    r, phi, s = f_t  # the sensor measurement you actually got
    # rhat is the measurement you should get (predicted readings)
    x, y, theta = x_t
    sigma_r, sigma_phi = 0.45, 0.45 # noise params associate with sensor readings

    subject = barcode_to_subject[s]
    landmark_x, landmark_y = subject_to_landmark[subject]

    # error between what you measured the dist to landmark to be vs. what it actually is
    r_hat = math.sqrt((landmark_x - x)**2 + (landmark_y - y)**2) 
    phi_hat = math.atan2(landmark_y - y, landmark_x - x) - theta   
    phi_error = normalize_angle(phi - phi_hat)
    q = prob(r - r_hat, sigma_r **2 ) * prob(phi_error, sigma_phi ** 2)
    # probability term for signature error is not included, as it will always be max probability (0 error) 
    # since we know the barcodes of landmarks w/ certainty
    return q

# simple particle filter implementation
#    ***IS NOT CALLED IN THIS PROGRAM *** 
# as we have to separate the prediction step and correction step between the staggered timesteps in the data
# these lines are adapted inside the main function
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
    # running control updates w/o filter (Q3)
    alpha_params = (0,0,0,0,0,0) # perfect path (no noise)
    initial_pose = df_groundtruth.iloc[0] # get initial pose from groundtruth file
    initial_pose = (initial_pose['x'], initial_pose['y'], initial_pose['orientation'])

    path_history2 = [initial_pose] # save poses for later plotting
    current_pose = initial_pose

    for i in range(len(df_control) -1):
        curr_t = df_control.iloc[i]["time"]
        next_t = df_control.iloc[i+1]["time"]
        dt = next_t-curr_t # calc the timestep

        v = df_control.iloc[i]["forward velocity"]
        w = df_control.iloc[i]["angular velocity"]

        current_pose = motion_model((v, w), current_pose,dt, alpha_params) # use motion model only
        path_history2.append(current_pose) # save for later

    # Extract x and y coordinates from the path history
    x_coords = [p[0] for p in path_history2]
    y_coords = [p[1] for p in path_history2]

    plt.figure()
    plt.plot(x_coords, y_coords, color ='green',linestyle='-', linewidth = 2, label='Robot Path')
        
    # Plot the ground truth path in blue
    plt.plot(df_groundtruth['x'], df_groundtruth['y'], color='blue', linestyle='-', linewidth=2, label='Ground Truth Path')
    
    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='Simulation End')
    plt.plot(df_groundtruth.iloc[-1]['x'], df_groundtruth.iloc[-1]['y'], 'bo', markersize=10, label='Ground Truth End')
    
    plt.title('Simulated vs. Ground Truth Robot Path from Motion Model Only')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensures the scaling of x and y axes are the same
    plt.show()


def sequence(alphas, title):
    # execute the sequence of commands given in Q2
    alpha_params = alphas # perfect path (no noise)
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
        current_pose = motion_model((v, w), current_pose, t,alpha_params)
        path_history.append(current_pose)
        
    # Extract x and y coordinates from the path history
    x_coords = [p[0] for p in path_history]
    y_coords = [p[1] for p in path_history]

    plt.figure()
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', label='Robot Path')
    
    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    plt.title(f'Robot Path from Motion Model {title}')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensures the scaling of x and y axes are the same
    plt.show()


# reports predicted range and bearing values (Q6)
def predict_test():

    def predict_range_heading(x_t, s):
        x, y, theta = x_t

        # df.loc[rows,cols] is pandas local based lookup
        # rows = mask -> keeps only rows where mask condition is true
        # columns = "some col name" -> returns values from that column
        subject = s

        landmark_x = df_landmarks.loc[df_landmarks["subject"] == subject, "x"].item()
        landmark_y = df_landmarks.loc[df_landmarks["subject"] == subject, "y"].item()

        # error between what you measured the dist to landmark to be vs. what it actually is
        r_hat = math.sqrt((landmark_x - x)**2 + (landmark_y - y)**2) #+ sample(sigma_r ** 2)
        phi_hat = math.atan2(landmark_y - y, landmark_x - x) - theta #+ sample(sigma_phi ** 2)

        return r_hat, phi_hat
    
    # table values from q6
    posn1 = (2,3,0)
    posn2 = (0,3,0)
    posn3 = (1,-2,0)

    # associated landmark subject #s from table in q6
    s1 = 6
    s2 = 13
    s3 = 17

    range1, bearing1 = predict_range_heading(posn1, s1) # range and bearing sensor values you should get when at specified positions for specified landmarks
    range2, bearing2 = predict_range_heading(posn2, s2)
    range3, bearing3 = predict_range_heading(posn3, s3)

    print(f"\nLandmark #6 Prediction at robot pose: {posn1}")
    print(f"Predicted measurement model values:\nRange: {range1}, Bearing: {bearing1}\n")
       
    print(f"\nLandmark #13 Prediction at robot pose: {posn2}")
    print(f"Predicted measurement model values:\nRange: {range2}, Bearing: {bearing2}\n")

    print(f"\nLandmark #17 Prediction at robot pose: {posn3}")
    print(f"Predicted measurement model values:\nRange: {range3}, Bearing: {bearing3}\n")


def sequence_comp():
    # compare sequence using full filter
    num_particles = 500
    alpha_params = (0.25,0.25,0.25,0.25,0.25,0.25) # perfect path (no noise)
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
    particles = [initial_pose] * num_particles # all particles to start are known to be at groundtruth location
    mean_pose_history = [initial_pose]

    for v, w, t in commands:
        particles = particle_filter(particles, (v, w),[], t,alpha_params)
        mean_x = np.mean([p[0] for p in particles])
        mean_y = np.mean([p[1] for p in particles])
        mean_theta = circular_mean([p[2] for p in particles])
        mean_pose_history.append((mean_x, mean_y, mean_theta))
        
    # Extract x and y coordinates from the path history
    x_coords = [p[0] for p in mean_pose_history]
    y_coords = [p[1] for p in mean_pose_history]

    plt.figure()
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', label='Robot Path')
    
    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    plt.title('Simulated Robot Path Sequence from Particle Filter')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensures the scaling of x and y axes are the same
    plt.show()

if __name__ == '__main__':
    sequence((0,0,0,0,0,0),  "(No Noise)")
    sequence((0.25,0.25,0.25,0.25,0.25,0.25), "With Noise")
    sequence_comp()
    predict_test()
    no_filter()

    barcode_to_subject, subject_to_landmark = create_lookup_tables(df_barcodes, df_landmarks)

    # Due to the staggered time steps in the measurement.dat and control.dat, best way to process is combine both dfs
    # add a 'type' column to each dataframe to identify the event type
    df_control['type'] = 'control'
    df_measurements['type'] = 'measurement'

    # combine the two dataframes into a single timeline for processing
    all_events = pd.concat([
        df_control[['time', 'forward velocity', 'angular velocity', 'type']],
        df_measurements[['time', 'barcode', 'range', 'bearing', 'type']]
    ])

    ignored_barcodes = {5, 14, 41, 32, 23} # set of barcodes for other 5 moving robots. these are not landmarks and should be ignored.

    # Sort to create the chronological timeline. "ignore index" relabels all rows in new df
    # edge case, if there are measurement rows and a control row at the same time, make sure they are ordered 
    # together so we do not accidentally skip a control 
    all_events = all_events.sort_values(by=['time','type'], ignore_index=True)

    # display first few events
    # print(all_events.head(10))

    # initialization
    num_particles = 750
    alpha_params = (0.75,0.75,0.75,0.75,0.75,0.75)

    initial_pose_df = df_groundtruth.iloc[0] # known init pose assumption
    initial_pose = (initial_pose_df['x'], initial_pose_df['y'], initial_pose_df['orientation'])
    particles = [initial_pose] * num_particles # all particles to start are known to be at groundtruth location
    mean_pose_history = [initial_pose] # init mean pose tracking for plotting later
    
    # first time from the first event in our sorted list (will be a control)
    last_update_time = all_events.iloc[0]['time'] 
    # Assume robot is stationary at the beginning
    last_known_control = (0.0, 0.0) 

    i = 0
    while i < len(all_events): # iterate through all rows in combined df
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
                    if barcode in ignored_barcodes: # ignore other robots
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
                    weights = [1.0 / num_particles] * num_particles # all particles will have uniform distribution
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
        mean_pose_history.append((mean_x, mean_y, mean_theta))

    plt.figure()
    x_coords = [p[0] for p in mean_pose_history]
    y_coords = [p[1] for p in mean_pose_history]
    plt.plot(x_coords, y_coords, color ='green',linestyle='-', linewidth = 2, label='Robot Path')
        
    # Plot the ground truth path in blue
    plt.plot(df_groundtruth['x'], df_groundtruth['y'], color='blue', linestyle='-', linewidth=2, label='Ground Truth Path')
    

    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='Simulation End')
    plt.plot(df_groundtruth.iloc[-1]['x'], df_groundtruth.iloc[-1]['y'], 'bo', markersize=10, label='Ground Truth End')
    
    plt.title('Simulated vs. Ground Truth Robot Path Using Particle Filter')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensures the scaling of x and y axes are the same
    plt.show()




import math
import random
import pandas as pd

# Read the .dat file
# delim_whitespace deprecated and replaced by sep
# \s+ means 1+ occurrences of whitespace is delimiter
df_barcodes = pd.read_csv("ds1/ds1_Barcodes.dat",sep=r'\s+',comment="#",header=None,names=["subject", "barcode"])
df_measurements = pd.read_csv("ds1/ds1_Measurement.dat",sep=r'\s+',comment="#",header=None,names=["time", "barcode", "range", "bearing"])
df_landmarks = pd.read_csv("ds1/ds1_Landmark_Groundtruth.dat",sep=r'\s+',comment="#",header=None,names=["subject", "x", "y","x_std_dev","y_std_dev"])


def sample(sigma_squared):
    # algo for sampling from approx normal distribution w/
    # zero mean and variance sigma^2

    summation = 0
    sigma = math.sqrt(sigma_squared) 

    for i in range(12):
        summation += random.uniform(-sigma, sigma)

    return 0.5 * summation

def predict_range_heading(x_t, s):
    x, y, theta = x_t

    sigma_r, sigma_phi = 0.1,0.1


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

if __name__ == '__main__':

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

    # all rows in dataset where landmark subject 6 was detected
    df_L6 = df_measurements[df_measurements["barcode"]== 63]
    df_L13 = df_measurements[df_measurements["barcode"]== 9]
    df_L17 = df_measurements[df_measurements["barcode"]== 54]

    # couple test points for detection of landmark row 1 (subject 6)
    count = 0
    print(f"\nLandmark #6 Prediction at robot pose: {posn1}")
    print(f"Predicted measurement model values:\nRange: {range1}, Bearing: {bearing1}\n")
    # for i in range(0,150,50):
    #     # i = 0, 50, 100 (3 iterations)
    #     row = df_L6.iloc[i] # skip 50 rows 
    #     # skip over similar readings at similar time steps to get a different location
    #     # input(row)
    #     measured_range = row["range"]
    #     measured_bearing = row["bearing"]
    #     time = row["time"]
    #     print(f"Actual sensor measurement values at time: {time}. Range: {measured_range}, Bearing: {measured_bearing}")
    #     print(f"Range reading error: {abs(range1 - measured_range)}, Bearing reading error: {abs(bearing1 - measured_bearing )}")
        
       
    print(f"\nLandmark #13 Prediction at robot pose: {posn2}")
    print(f"Predicted measurement model values:\nRange: {range2}, Bearing: {bearing2}\n")
    # couple test points for detection of landmark row 2 (subject 13)
    # for i in range(0,150,50):
    #     row = df_L13.iloc[i] # skip 50 rows 

    #     measured_range = row["range"]
    #     measured_bearing = row["bearing"]
    #     time = row["time"]
        
    #     print(f"Actual sensor measurement values at time: {time}. Range: {measured_range}, Bearing: {measured_bearing}")
    #     print(f"Range reading error: {abs(range2 - measured_range)}, Bearing reading error: {abs(bearing2 - measured_bearing )}")
       
       

    # couple test points for detection of landmark row 3 (subject 17)
    print(f"\nLandmark #17 Prediction at robot pose: {posn3}")
    print(f"Predicted measurement model values:\nRange: {range3}, Bearing: {bearing3}\n")

    # for i in range(0,150,50):
    #     row = df_L17.iloc[i] # skip 50 rows 
    #     measured_range = row["range"]
    #     measured_bearing = row["bearing"]
    #     time = row["time"]
    #     print(f"Actual sensor measurement values at time: {time}. Range: {measured_range}, Bearing: {measured_bearing}")
    #     print(f"Range reading error: {abs(range3 - measured_range)}, Bearing reading error: {abs(bearing3 - measured_bearing )}")

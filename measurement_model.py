import math
import pandas as pd

# Read the .dat file
df_barcodes = pd.read_csv("ds1/ds1_Barcodes.dat",delim_whitespace=True,comment="#",header=None,names=["subject", "barcode"])
df_measurements = pd.read_csv("ds1/ds1_Measurement.dat",delim_whitespace=True,comment="#",header=None,names=["time", "barcode", "range", "bearing"])
df_landmarks = pd.read_csv("ds1/ds1_Landmark_Groundtruth.dat",delim_whitespace=True,comment="#",header=None,names=["subject", "x", "y","x_std_dev","y_std_dev"])

# print(df.head())

def prob(a,b2):
    # computes probability of its arg a under a zero centered normal
    # distribution w/ variance b^2 
    return math.exp(-(a**2)/(2*b2)) / (math.sqrt(2 * math.pi * b2))

# input current robot pose
# input measurement data from df 
def landmark_model_known_correspondence(f_t,x_t):
    # f at time t : observed feature f = (r, phi, s). Where you measured the landmark to be.
    # c at time t : true identity of the feature c ... subject #
    # x at time t : robot pose x = (x y Θ)
    # map m

    r, phi, s = f_t  # the sensor measurement you got
    # rhat is the measurement you should get
    x, y, theta = x_t
    sigma_r, sigma_phi = 0.1,0.1


    # df.loc[rows,cols] is label based lookup
    # rows = mask -> keeps only rows where mask condition is true
    # columns = "some col name" -> returns values from that column
    # .item() extract single value (works b/c we are only expecting 1 element)
    subject = df_barcodes.loc[df_barcodes["barcode"] == s, "subject"].item() 

    landmark_x = df_landmarks.loc[df_landmarks["subject"] == subject, "x"].item()
    landmark_y = df_landmarks.loc[df_landmarks["subject"] == subject, "y"].item()

    # error between what you measured the dist to landmark to be vs. what it actually is
    r_hat = math.sqrt((landmark_x - x)**2 + (landmark_y - y)**2) 
    phi_hat = math.atan2(landmark_y - y, landmark_x - x) - theta   
    q = prob(r - r_hat, sigma_r **2 ) * prob(phi - phi_hat, sigma_phi ** 2)
    # probability term for signature error is not included, as it will always be max probability (0 error) 
    # since we know the barcodes of landmarks w/ certainty

    return q


if __name__ == '__main__':

    landmark_model_known_correspondence()
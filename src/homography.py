import cv2
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

def gauss(t, r, window_size):
    """
    @param: window_size is the size of window over which gaussian to be applied
    @param: t is the index of current point 
    @param: r is the index of point in window 
    
    Return:
            returns spacial guassian weights over a window size
    """
    return np.exp((-9*(r-t)**2)/window_size**2)

def optimize_path(c, buffer_size=0, window_size=10):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term
    
    
    Returns:
            returns an optimized gaussian smooth camera trajectory 
    """
    lambda_t = 100
    if window_size > c.shape[0]:
        window_size = c.shape[0]

    P = Variable(c.shape[0])
    for t in range(c.shape[0]):
        # first term for optimised path to be close to camera path
        path_term = (P[t]-c[t])**2

        # second term for smoothness using gaussian weights
        for r in range(window_size):
            if t-r < 0:
                break
            w = gauss(t, t-r, window_size)
            gauss_weight = w*(P[t]-P[t-r])**2
            if r == 0:
                gauss_term = gauss_weight
            else:
                gauss_term += gauss_weight

        if t == 0:
            objective = path_term + lambda_t*gauss_term
        else:
            objective += path_term + lambda_t*gauss_term

    prob = Problem(Minimize(objective))
    prob.solve()
    return np.asarray(P.value)


if __name__ == '__main__':

    filename = '../../data/test/shaky-4.avi'
    cap = cv2.VideoCapture(filename)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # generate stabilized video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('stable-4.avi', fourcc, frame_rate, (2*frame_width, frame_height))

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # preserve aspect ratio
    HORIZONTAL_BORDER = 50
    VERTICAL_BORDER = (HORIZONTAL_BORDER*old_gray.shape[1])/old_gray.shape[0]

    print '--Generation--'
    frame_num = 0
    prev_motion = []
    while frame_num < frame_count-2:
        try:
            # processing frames
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find corners in it
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # estimate motion mesh for old_frame
            H = cv2.estimateRigidTransform(good_old, good_new, False)

            try:
                dx = H[0, 2]
            except:
                dx = prev_motion[-1][0]
            try:
                dy = H[1, 2]
            except:
                dy = prev_motion[-1][1]
            try:
                da = np.arctan2(H[1, 0], H[0, 0])
            except:
                da = prev_motion[-1][2]
            prev_motion.append([dx, dy, da])

            # updates frames
            frame_num += 1
            old_frame = frame.copy()
            old_gray = frame_gray.copy()
        except:
            break

    print '--Optimization--'
    x = 0; x_path = [];
    y = 0; y_path = [];
    a = 0; a_path = [];
    for i in range(len(prev_motion)):
        x += prev_motion[i][0]
        x_path.append(x)

        y += prev_motion[i][1]
        y_path.append(y)

        a += prev_motion[i][2]
        a_path.append(a)

    smooth_x = optimize_path(np.asarray(x_path))
    smooth_y = optimize_path(np.asarray(y_path))
    smooth_a = optimize_path(np.asarray(a_path))

    plt.plot(np.asarray(x_path))
    plt.plot(np.asarray(smooth_x))
    plt.savefig('stable-x.png')
    plt.clf()

    plt.plot(np.asarray(y_path))
    plt.plot(np.asarray(smooth_y))
    plt.savefig('stable-y.png')
    plt.clf()

    plt.plot(np.asarray(a_path))
    plt.plot(np.asarray(smooth_a))
    plt.savefig('stable-a.png')
    plt.clf()
    
    smooth_motion = []
    for i in range(len(prev_motion)):
        smooth_motion.append([prev_motion[i][0]+smooth_x[i]-x_path[i], 
                              prev_motion[i][1]+smooth_y[i]-y_path[i],
                              prev_motion[i][2]+smooth_a[i]-a_path[i]])
    
    print '--Reconstruction--'
    frame_num = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while frame_num < frame_count-3:
        try:
            ret, frame = cap.read()
            dx, dy, da = smooth_motion[frame_num]
            
            T = np.zeros((2, 3))
            T[0, 0] = np.cos(da); T[0, 1] = -np.sin(da); T[0, 2] = dx;
            T[1, 0] = np.sin(da); T[1, 1] = np.cos(da); T[1, 2] = dy;

            new_frame = cv2.warpAffine(frame, T, (frame.shape[1], frame.shape[0]))
            # cv2.imwrite('videos/'+str(frame_num)+'.png', new_frame)
            new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
            new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            output = np.concatenate((frame, new_frame), axis=1)
            out.write(output)
            frame_num += 1
        except:
            break

    cap.release()
    out.release()

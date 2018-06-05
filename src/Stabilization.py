import sys
import cv2
import numpy as np
from Optimization import optimize_path
from MeshFlow import motion_propagate
from MeshFlow import mesh_warp_frame
from MeshFlow import generate_vertex_profiles

# block of size in mesh
PIXELS = 16

# motion propogation radius
RADIUS = 300

@measure_performance
def read_video(cap):
    """
    @param: cap is the cv2.VideoCapture object that is
            instantiated with given video

    Returns:
            returns mesh vertex motion vectors & 
            mesh vertex profiles 
    """

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 1000,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    # Take first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # preserve aspect ratio
    global HORIZONTAL_BORDER
    HORIZONTAL_BORDER = 30

    global VERTICAL_BORDER
    VERTICAL_BORDER = (HORIZONTAL_BORDER*old_gray.shape[1])/old_gray.shape[0]

    # motion meshes in x-direction and y-direction
    x_motion_meshes = []; y_motion_meshes = []

    # path parameters
    x_paths = np.zeros((old_frame.shape[0]/PIXELS, old_frame.shape[1]/PIXELS, 1))
    y_paths = np.zeros((old_frame.shape[0]/PIXELS, old_frame.shape[1]/PIXELS, 1))

    frame_num = 1
    bar = tqdm(total=frame_count)
    while frame_num < frame_count:

        # processing frames
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find corners in it
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # estimate motion mesh for old_frame
        x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
        try:
            x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
            y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)
        except:
            x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
            y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)

        # generate vertex profiles
        x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

        # updates frames
        bar.update(1)
        frame_num += 1
        old_frame = frame.copy()
        old_gray = frame_gray.copy()

    bar.close()
    return x_motion_meshes, y_motion_meshes, x_paths, y_paths


@measure_performance
def stabilize(x_paths, y_paths):
    """
    @param: x_paths is motion vector accumulation on 
            mesh vertices in x-direction
    @param: y_paths is motion vector accumulation on
            mesh vertices in y-direction

    Returns:
            returns optimized mesh vertex profiles in
            x-direction & y-direction
    """

    # optimize for smooth vertex profiles
    sx_paths = optimize_path(x_paths)
    sy_paths = optimize_path(y_paths)
    return sx_paths, sy_paths


def plot_vertex_profiles(x_paths, sx_paths):
    """
    @param: x_paths is original mesh vertex profiles
    @param: sx_paths is optimized mesh vertex profiles

    Return:
            saves equally spaced mesh vertex profiles
            in directory '<PWD>/results/'
    """

    # plot some vertex profiles
    for i in range(0, x_paths.shape[0]):
        for j in range(0, x_paths.shape[1], 10):
            plt.plot(x_paths[i, j, :])
            plt.plot(sx_paths[i, j, :])
            plt.savefig('results/'+str(i)+'_'+str(j)+'.png')
            plt.clf()


def get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths):
    """
    @param: x_motion_meshes is the motion vectors on
            mesh vertices in x-direction
    @param: y_motion_meshes is the motion vectors on
            mesh vertices in y-direction
    @param: x_paths is motion vector accumulation on 
            mesh vertices in x-direction
    @param: y_paths is motion vector accumulation on
            mesh vertices in y-direction    
    @param: sx_paths is the optimized motion vector
            accumulation in x-direction
    @param: sx_paths is the optimized motion vector
            accumulation in x-direction

    Returns:
            returns a update motion mesh for each frame
            with which that needs to be warped
    """

    # U = P-C
    x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_meshes[:, :, -1], axis=2)), axis=2)
    y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_meshes[:, :, -1], axis=2)), axis=2)
    new_x_motion_meshes = sx_paths-x_paths
    new_y_motion_meshes = sy_paths-y_paths
    return x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes


@measure_performance
def generate_stabilized_video(cap, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes):
    """
    @param: cap is the cv2.VideoCapture object that is
            instantiated with given video
    @param: x_motion_meshes is the motion vectors on
            mesh vertices in x-direction
    @param: y_motion_meshes is the motion vectors on
            mesh vertices in y-direction
    @param: new_x_motion_meshes is the updated motion vectors 
            on mesh vertices in x-direction to be warped with
    @param: new_y_motion_meshes is the updated motion vectors 
            on mesh vertices in y-direction to be warped with
    """
    
    # get video properties
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # generate stabilized video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('stable.avi', fourcc, frame_rate, (2*frame_width, frame_height))

    bar = tqdm(total=frame_count)
    while frame_num < frame_count:
        try:
            # reconstruct from frames
            ret, frame = cap.read()
            x_motion_mesh = x_motion_meshes[:, :, frame_num]
            y_motion_mesh = y_motion_meshes[:, :, frame_num]
            new_x_motion_mesh = new_x_motion_meshes[:, :, frame_num]
            new_y_motion_mesh = new_y_motion_meshes[:, :, frame_num]
            
            # mesh warping
            new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
            new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
            new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            output = np.concatenate((frame, new_frame), axis=1)
            out.write(output)
            
            # draw old motion vectors
            r = 5
            for i in range(x_motion_mesh.shape[0]):
                for j in range(x_motion_mesh.shape[1]):
                    theta = np.arctan2(y_motion_mesh[i, j], x_motion_mesh[i, j])
                    cv2.line(frame, (j*PIXELS, i*PIXELS), (int(j*PIXELS+r*np.cos(theta)), int(i*PIXELS+r*np.sin(theta))), 1)
            cv2.imwrite('results/old_motion_vectors/'+str(frame_num)+'.jpg', frame)

            # draw new motion vectors
            for i in range(new_x_motion_mesh.shape[0]):
                for j in range(new_x_motion_mesh.shape[1]):
                    theta = np.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
                    cv2.line(new_frame, (j*PIXELS, i*PIXELS), (int(j*PIXELS+r*np.cos(theta)), int(i*PIXELS+r*np.sin(theta))), 1)
            cv2.imwrite('results/new_motion_vectors/'+str(frame_num)+'.jpg', new_frame)

            frame_num += 1
            bar.update(1)
        except:
            break
    
    bar.close()
    cap.release()
    out.release()


if __name__ == '__main__':
    
    # measure time required
    start_time = time.time()

    # get video properties
    file_name = sys.argv[1]
    cap = cv2.VideoCapture(file_name)
    
    # propogate motion vectors and generate vertex profiles
    x_motion_meshes, y_motion_meshes, x_paths, y_paths = read_video(cap)
    
    # stabilize the vertex profiles
    sx_paths, sy_paths = stabilize(x_paths, y_paths)
    
    # visualize optimized paths
    plot_vertex_profiles(x_paths, sx_paths)

    # get updated mesh warps
    x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes = get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths)

    # apply updated mesh warps & save the result
    generate_stabilized_video(cap, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes)

    print 'Time elapsed: ', str(time()-start_time)
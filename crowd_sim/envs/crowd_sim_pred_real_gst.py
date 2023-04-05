import gym
import numpy as np
from dr_spaam.detector import Detector
import cv2
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from crowd_sim.envs.crowd_sim_pred import CrowdSimPred
import matplotlib.pyplot as plt



# dr_model_file = 'trained_models/ckpt_jrdb_ann_ft_dr_spaam_e20.pth'
dr_model_file = 'trained_models_lidar/ckpt_jrdb_ann_drow3_e40.pth'

class CrowdSimPredRealGST(CrowdSimPred):
    '''
    Same as CrowdSimPred, except that
    The future human traj in 'spatial_edges' are dummy placeholders
    and will be replaced by the outputs of a real GST pred model in the wrapper function in vec_pretext_normalize.py
    '''
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super(CrowdSimPredRealGST, self).__init__()
        self.pred_method = None

        # to receive data from gst pred model
        self.gst_out_traj = None

        #lidar params 
        self.lidar_fov = 360
        self.scan_points = 450
        self.lidar_resolution = np.deg2rad(self.lidar_fov/self.scan_points)
        self.detector = Detector(dr_model_file, model='DROW3',gpu=True,stride=1,panoramic_scan=True)
        self.detector.set_laser_fov(self.lidar_fov)
        self.detections = []
        self.use_detections = True
        self.add_noise = False
        self.scans = []
        self.detections_xy = []
        self.noisy  = []

    def scan_lidar(self, noise = 0.3):
        # get scan as a dictionary {angle_index : distance}
        res = self.scan_points
        full_scan = {}
        for h in self.humans:
            scan = h.get_scan(res, self.robot.px, self.robot.py)
            for angle in scan:
                if scan[angle] < full_scan.get(angle, np.inf):
                    full_scan[angle] = scan[angle]

        # convert to array of length res, with inf at angles with no reading
        out_scan = np.zeros(res) + np.inf
        for k in full_scan.keys():
            out_scan[k] = full_scan[k]

        out_scan = out_scan + noise*np.random.random(len(out_scan)) - 0.2
        dim = 6
        x = self.robot.px
        y = self.robot.py
        theta1 = np.arctan((dim-y)/(dim-x))
        theta2 = np.pi - np.arctan((dim-y)/(dim+x))
        theta3 = 1.5*np.pi - np.arctan((dim+x)/(dim+y))
        theta4 = 1.5*np.pi + np.arctan((dim-x)/(dim+y))

        for i in range(res):
            if out_scan[i] == np.inf:
                out_scan[i] = 200 # todo: maybe add some rigour here
        return out_scan
        
    def find_assignment(self, prev_dets_xy, dets_xy):

        # use hungarian algorithm to find correspondences between current and previous detections
        n = self.human_num
        m = len(dets_xy)
        if m > n: raise ValueError("Number of detections must not be greater than the number of humans")
        
        # compute cost matrix
        costs = np.zeros((n, m))
    
        for i in range(n):
            for j in range(m):
                dist = distance.euclidean(prev_dets_xy[i], dets_xy[j])
                costs[i][j] = dist

        # compute optimal assignments
        row_ind, col_ind = linear_sum_assignment(costs)

        return row_ind, col_ind
    
    def get_detections(self, scan):
        # get people detections (positions)
        full_dets_xy, dets_cls, instance_mask = self.detect(scan) 
        # print(len(full_dets_xy))

        # determine most likely detections 
        detect_idx = np.argsort(dets_cls)
        dets_xy = full_dets_xy[detect_idx]
        if len(dets_xy) > self.human_num: dets_xy = dets_xy[-self.human_num:]

        # process detections
        robot_positions = [self.robot.px, self.robot.py]
        dets_xy *= -1
        dets_xy += robot_positions[-1]

        # flag for detections < human_num in first step
        if len(self.detections) == 0:
            self.max_detections = len(dets_xy)
        else:
            self.max_detections = max(self.max_detections, len(dets_xy))

        if self.max_detections < self.human_num:
            for i in range(self.human_num - self.max_detections):
                dets_xy = np.append(dets_xy, [dets_xy[-1]], axis=0)


        if len(self.detections) > 0: # beyond first pass - match to previous detections
            # get previous set of detections
            prev_dets_xy = self.detections[-1]

            # compute assignments and changing of detection arrays
            rows, cols = self.find_assignment(prev_dets_xy, dets_xy)
            new_dets = np.zeros((self.human_num, 2))
            for i in range(len(cols)):
                new_dets[rows[i]] = dets_xy[cols[i]]
            for i in list(set(range(self.human_num)) - set(rows)):
                new_dets[i] = prev_dets_xy[i]

            dets_xy = new_dets.copy()
            

        assert len(dets_xy) == len(self.humans)
        self.detections.append(dets_xy)

        for i, pos in enumerate(dets_xy):
            self.humans[i].set_detected_state(pos, self.step_counter)

        return dets_xy

    def detect(self, scan):
        dets, dcls, mask = self.detector(scan)
        del_mask = np.where(np.linalg.norm(dets, axis=-1) > 10, True, False)
        dcls = np.delete(dcls, del_mask, axis=0)
        dets = np.delete(dets, del_mask, axis=0)
        # print(len(dets))
            
        return dets, dcls, mask
    
    def scan_to_points(self, scan):
        coords = []
        for i in range(len(scan)):
            ang = 360*i/self.scan_points
            coords.append([self.robot.px + scan[i]*np.cos(np.deg2rad(ang)), self.robot.py + scan[i]*np.sin(np.deg2rad(ang))])

        return coords


    def set_robot(self, robot):
        """set observation space and action space"""
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d = {}
        # robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        '''
        format of spatial_edges: [max_human_num, [state_t, state_(t+1), ..., state(t+self.pred_steps)]]
        '''

        # predictions only include mu_x, mu_y (or px, py)
        self.spatial_edge_dim = int(2*(self.predict_steps+1))

        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(self.config.sim.human_num + self.config.sim.human_num_range, self.spatial_edge_dim),
                            dtype=np.float32)

        # masks for gst pred model
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.human_num + self.config.sim.human_num_range,),
                                            dtype=np.bool)

        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def talk2Env(self, data):
        """
        Call this function when you want extra information to send to/recv from the env
        :param data: data that is sent from gst_predictor network to the env, it has 2 parts:
        output predicted traj and output masks
        :return: True means received
        """
        self.gst_out_traj=data
        return True


    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset, sort=False):
        """Generate observation for reset and step functions"""
        # since gst pred model needs ID tracking, don't sort all humans
        # inherit from crowd_sim_lstm, not crowd_sim_pred to avoid computation of true pred!
        # sort=False because we will sort in wrapper in vec_pretext_normalize.py later
        parent_ob = super(CrowdSimPred, self).generate_ob(reset=reset, sort=False)

        # add additional keys, removed unused keys
        ob = {}

        ob['visible_masks'] = parent_ob['visible_masks']
        ob['robot_node'] = parent_ob['robot_node']
        ob['temporal_edges'] = parent_ob['temporal_edges']

        ob['spatial_edges'] = np.tile(parent_ob['spatial_edges'], self.predict_steps+1)

        ob['detected_human_num'] = parent_ob['detected_human_num']

        # Use scan for observation 
        scan = self.scan_lidar()
        self.scans.append(scan)
        detection_xy = self.get_detections(scan)
        self.detections_xy.append(detection_xy)

        if self.use_detections:
            ob['spatial_edges'][:,:2] = detection_xy
        elif self.add_noise:
            ob_noise = ob['spatial_edges'][:,:2] + 0.2*np.random.randn(self.human_num, 2) 
            ob['spatial_edges'][:,:2] = ob_noise
            self.noisy.append(ob_noise)


        return ob


    def calc_reward(self, action, danger_zone='future'):
        # inherit from crowd_sim_lstm, not crowd_sim_pred to prevent social reward calculation
        # since we don't know the GST predictions yet
        reward, done, episode_info = super(CrowdSimPred, self).calc_reward(action, danger_zone=danger_zone)
        return reward, done, episode_info


    def render(self, mode='human'):
        """
        render function
        use talk2env to plot the predicted future traj of humans
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)
        
        # add lidar scans 
        scan_xy = self.scan_to_points(self.scans[-1])
        xs = [scan_xy[i][0] for i in range(len(scan_xy))]
        ys = [scan_xy[i][1] for i in range(len(scan_xy))]
        
        scatter = ax.scatter(xs, ys, c ='b', s=6)

        # add detections
        if self.use_detections:
            detection_xy = self.detections_xy[self.step_counter]
            # detection_xy = np.array([[human.detector_px, human.detector_py] for human in self.humans]) 
        elif self.add_noise:
            detection_xy = self.noisy[-1]
        else:
            detection_xy = np.array([[human.px, human.py] for human in self.humans]) 


        # detection_xy = np.array([[human.px, human.py] for human in self.humans]) # + 0.3*np.random.random((self.human_num, 2))
        # detection_xy = self.detections[-1]

        xs = [detection_xy[i][0] for i in range(len(detection_xy))]
        ys = [detection_xy[i][1] for i in range(len(detection_xy))]

        scatter2 = ax.scatter(xs, ys, c='r', s=70, edgecolors='k')

        # add detection labels
        alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        det_human_numbers = [
            plt.text(detection_xy[i][0] +.5,
                    detection_xy[i][1]  + .5,
                    str(alph[i]),
                    color='r',
                    fontsize=12) for i in range(len(detection_xy))
        ]

        for i in range(len(det_human_numbers)):
            ax.add_artist(det_human_numbers[i])

        # draw FOV for the robot
        # add robot FOV
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range + self.robot.radius+self.config.humans.radius, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.humans]

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5

        # plot the current human states
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.human_visibility[i]:
                human_circles[i].set_color(c='b')
            else:
                human_circles[i].set_color(c='r')

            if -actual_arena_size <= self.humans[i].px <= actual_arena_size and -actual_arena_size <= self.humans[
                i].py <= actual_arena_size:
                # label numbers on each human
                # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
                plt.text(self.humans[i].px , self.humans[i].py , i, color='black', fontsize=12)

        # plot predicted human positions
        for i in range(len(self.humans)):
            # add future predicted positions of each human
            if self.gst_out_traj is not None:
                for j in range(self.predict_steps):
                    circle = plt.Circle(self.gst_out_traj[i, (2 * j):(2 * j + 2)] + np.array([robotX, robotY]),
                                        self.config.humans.radius, fill=False, color='tab:orange', linewidth=1.5)
                    # circle = plt.Circle(np.array([robotX, robotY]),
                    #                     self.humans[i].radius, fill=False)
                    ax.add_artist(circle)
                    artists.append(circle)

        plt.pause(0.1)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        scatter.remove()
        scatter2.remove()
        for t in ax.texts:
            t.set_visible(False)

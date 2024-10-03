import re

import numpy as np
import pygame
import math
import time
import pickle
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import socket
import libemg

class FittsLawTest:
    def __init__(self, num_circles=30, num_trials=15, savefile="out.pkl", logging=True, width=1250, height=750):
        pygame.init()
        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode([width, height])
        self.clock = pygame.time.Clock()
        
        # logging information
        self.log_dictionary = {
            'time_stamp':        [],
            'trial_number':      [],
            'goal_circle' :      [],
            'global_clock' :     [],
            'cursor_position':   [],
            'class_label':       [],
            'current_direction': []
        }

        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.YELLOW = (255,255,0)
        self.BLUE   = (0,102,204)
        self.small_rad = 40
        self.big_rad   = 275
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.done = False
        self.VEL = 25
        self.dwell_time = 3
        self.num_of_circles = num_circles 
        self.max_trial = num_trials
        self.width = width
        self.height = height
        # self.fps = 1/(config.window_increment / 200)
        self.fps = 60
        self.savefile = savefile
        self.logging = logging
        self.trial = 0

        # interface objects
        self.circles = []
        self.cursor = pygame.Rect(self.width//2 - 7, self.height//2 - 7, 14, 14)
        self.goal_circle = -1
        self.get_new_goal_circle()
        self.current_direction = [0,0]

        # timer for checking socket
        self.window_checkpoint = time.time()

        # Socket for reading EMG
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        self.sock.bind(('127.0.0.1', 12346))
        self.timeout_timer = None
        self.timeout = 30   # (seconds)
        self.trial_duration = 0

    def draw(self):
        self.screen.fill(self.BLACK)
        self.draw_circles()
        self.draw_cursor()
        self.draw_timer()
    
    def draw_circles(self):
        if not len(self.circles):
            self.angle = 0
            self.angle_increment = 360 // self.num_of_circles
            while self.angle < 360:
                self.circles.append(pygame.Rect((self.width//2 - self.small_rad) + math.cos(math.radians(self.angle)) * self.big_rad, (self.height//2 - self.small_rad) + math.sin(math.radians(self.angle)) * self.big_rad, self.small_rad * 2, self.small_rad * 2))
                self.angle += self.angle_increment

        for circle in self.circles:
            pygame.draw.circle(self.screen, self.RED, (circle.x + self.small_rad, circle.y + self.small_rad), self.small_rad, 2)
        
        goal_circle = self.circles[self.goal_circle]
        pygame.draw.circle(self.screen, self.RED, (goal_circle.x + self.small_rad, goal_circle.y + self.small_rad), self.small_rad)
            
    def draw_cursor(self):
        pygame.draw.circle(self.screen, self.YELLOW, (self.cursor.left + 7, self.cursor.top + 7), 7)

    def draw_timer(self):
        if hasattr(self, 'dwell_timer'):
            if self.dwell_timer is not None:
                toc = time.perf_counter()
                duration = round((toc-self.dwell_timer),2)
                time_str = str(duration)
                draw_text = self.font.render(time_str, 1, self.BLUE)
                self.screen.blit(draw_text, (10, 10))

    def update_game(self):
        self.draw()
        self.run_game_process()
        self.move()
    
    def run_game_process(self):
        self.check_collisions()
        self.check_events()

    def check_collisions(self):
        circle = self.circles[self.goal_circle]
        if math.sqrt((circle.centerx - self.cursor.centerx)**2 + (circle.centery - self.cursor.centery)**2) < (circle[2]/2 + self.cursor[2]/2):
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.goal_circle))
            self.Event_Flag = True
        else:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.num_of_circles))
            self.Event_Flag = False

    def check_events(self):
        # closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return
            
            
        data, _ = self.sock.recvfrom(1024)
        data = str(data.decode("utf-8"))
        self.window_checkpoint = time.time()
        
        self.current_direction = [0,0]
        if data:
            # Remove unneeded characters
            dof_values = parse_dof_values(data)
            self.current_direction[0] += self.VEL * float(dof_values[0])
            self.current_direction[1] -= self.VEL * float(dof_values[1])

            if self.logging:
                self.log(str(dof_values), time.time())
            
        

        ## CHECKING FOR COLLISION BETWEEN CURSOR AND RECTANGLES
        if event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + self.num_of_circles:
            if self.dwell_timer is None:
                self.dwell_timer = time.perf_counter()
            else:
                toc = time.perf_counter()
                self.duration = round((toc - self.dwell_timer), 2)
            if self.duration >= self.dwell_time:
                self.get_new_goal_circle()
                self.dwell_timer = None
                if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                    self.trial += 1
                else:
                    if self.logging:
                        self.save_log()
                    self.done = True
        elif event.type == pygame.USEREVENT + self.num_of_circles:
            if self.Event_Flag == False:
                self.dwell_timer = None
                self.duration = 0
        if self.timeout_timer is None:
            self.timeout_timer = time.perf_counter()
        else:
            toc = time.perf_counter()
            self.trial_duration = round((toc - self.timeout_timer), 2)
        if self.trial_duration >= self.timeout:
            # Timeout
            self.get_new_goal_circle()
            self.timeout_timer = None
            if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                self.trial += 1
            else:
                if self.logging:
                    self.save_log()
                self.done = True

    def move(self):
        # Making sure its within the bounds of the screen
        if self.cursor.left + self.current_direction[0] > 0 and self.cursor.left + self.current_direction[0] < self.width:
            self.cursor.left += self.current_direction[0]
        if self.cursor.top + self.current_direction[1] > 0 and self.cursor.top + self.current_direction[1] < self.height:
            self.cursor.top += self.current_direction[1]
    
    def get_new_goal_circle(self):
        if self.goal_circle == -1:
            self.goal_circle = 0
            self.next_circle_in = self.num_of_circles//2
            self.circle_jump = 0
        else:
            self.goal_circle =  (self.goal_circle + self.next_circle_in )% self.num_of_circles
            if self.circle_jump == 0:
                self.next_circle_in = self.num_of_circles//2 + 1
                self.circle_jump = 1
            else:
                self.next_circle_in = self.num_of_circles // 2
                self.circle_jump = 0
        self.timeout_timer = None
        self.trial_duration = 0


    def log(self, label, timestamp):
        circle = self.circles[self.goal_circle]
        self.log_dictionary['time_stamp'].append(timestamp)
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_circle'].append((circle.centerx, circle.centery, circle[2]))
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.centerx, self.cursor.centery, self.cursor[2]))
        self.log_dictionary['class_label'].append(label) 
        self.log_dictionary['current_direction'].append(self.current_direction)

    def save_log(self):
        # Adding timestamp
        with open(self.savefile, 'wb') as f:
            pickle.dump(self.log_dictionary, f)

    def run(self):
        while not self.done:
            # updated frequently for graphics & gameplay
            self.update_game()
            pygame.display.update()
            self.clock.tick(self.fps)
            pygame.display.set_caption(str(self.clock.get_fps()))
        pygame.quit()

def parse_dof_values(data_str):
    return libemg.emg_predictor.OnlineEMGRegressor.parse_output(data_str)[0]


def in_circle(cursor, circle):
    return math.sqrt((circle[0] - cursor[0])**2 + (circle[1] - cursor[1])**2) < circle[2]/2 + cursor[2]/2


def is_timeout_trial(trial, run_log):
    timeout_trials = find_timeout_trials(run_log)
    return trial in timeout_trials

def calculate_efficiency(run_log):
    efficiency = []
    trials = np.unique(run_log['trial_number'])
    for t in trials:
        if is_timeout_trial(t, run_log):
            # Ignore trial
            continue
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        distance_travelled = np.sum([math.dist(run_log['cursor_position'][t_idxs[i]][0:2], run_log['cursor_position'][t_idxs[i-1]][0:2]) for i in range(1,len(t_idxs))])
        fastest_path = math.dist((run_log['cursor_position'][t_idxs[0]])[0:2], (run_log['goal_circle'][t_idxs[0]])[0:2])
        efficiency.append(fastest_path/distance_travelled)
    return np.mean(efficiency)


def calculate_throughput(run_log):
    # See https://www.yorku.ca/mack/hhci2018.html for explanation of effective width and TP calculation
    trials = np.unique(run_log['trial_number'])
    effective_amplitudes = []
    effective_distances = []
    trial_times = []
    for t in trials:
        if is_timeout_trial(t, run_log):
            # Ignore trial
            continue
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        trial_time = run_log['global_clock'][t_idxs[-1]] - run_log['global_clock'][t_idxs[0]]
        trial_times.append(trial_time)
        starting_cursor_position = (run_log['cursor_position'][t_idxs[0]])[0:2]
        goal_circle_position = (run_log['goal_circle'][t_idxs[0]])[0:2]
        final_cursor_position = (run_log['cursor_position'][t_idxs[-1]])[0:2]  # maybe take the average across the dwell time if we have time
        a = math.dist(starting_cursor_position, goal_circle_position)
        b = math.dist(final_cursor_position, goal_circle_position)
        c = math.dist(starting_cursor_position, final_cursor_position)
        dx = (c * c - b * b - a * a) / (2 * a)
        ae = a + dx
        effective_amplitudes.append(ae)
        effective_distances.append(dx)

    if len(effective_amplitudes) < 2:
        # Return throughput of 0 if acquired less than 2 targets
        return 0
    run_effective_ampitude = np.mean(effective_amplitudes)
    run_effective_width = np.std(effective_distances) * 4.333
    effective_id = math.log2((run_effective_ampitude / run_effective_width) + 1)
    movement_time = np.mean(trial_times)

    return effective_id / movement_time


def calculate_overshoots(run_log):
    overshoots = []
    trials = np.unique(run_log['trial_number'])
    for t in trials:
        if is_timeout_trial(t, run_log):
            # Ignore trial
            continue
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        cursor_locs = np.array(run_log['cursor_position'])[t_idxs]
        targets = np.array(run_log['goal_circle'])[t_idxs]
        in_bounds = [in_circle(cursor_locs[i], targets[i]) for i in range(0,len(cursor_locs))]
        trial_overshoots = 0
        for i in range(1,len(in_bounds)):
            if in_bounds[i-1] == True and in_bounds[i] == False:
                trial_overshoots += 1 
        overshoots.append(trial_overshoots)
    return sum(overshoots)

def find_timeout_trials(run_log):
    trials = np.unique(run_log['trial_number'])
    timeout_trials = []
    for t in trials:
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        trial_time = run_log['global_clock'][t_idxs[-1]] - run_log['global_clock'][t_idxs[0]]
        if trial_time > 30:
            # Went to timeout
            timeout_trials.append(t)
    return timeout_trials


def extract_traces(run_log):
    trials = np.unique(run_log['trial_number'])
    traces = []
    for t in trials:
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        traces.append(np.array(run_log['cursor_position'])[t_idxs][:, :2])
    return traces


def extract_model_predictions(run_log):
    trials = np.unique(run_log['trial_number'])
    predictions = []
    for t in trials:
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        model_output = np.array(run_log['class_label'])[t_idxs]
        model_output = np.array(list(map(parse_dof_values, model_output)))
        predictions.append(model_output)
    return predictions


def extract_fitts_metrics(run_log):
    fitts_results = {}
    fitts_results['timeouts'] = find_timeout_trials(run_log)
    fitts_results['overshoots'] = calculate_overshoots(run_log)
    fitts_results['efficiency'] = calculate_efficiency(run_log)
    fitts_results['throughput'] = calculate_throughput(run_log)
    return fitts_results


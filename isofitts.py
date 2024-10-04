import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import pygame
import math
import time
import pickle
import os
import socket
from config import Config
config = Config()
from utils import setup_live_processing
from models import setup_classifier
import numpy as np
from multiprocessing import Lock

save_location = f"Data/subject{config.subjectID}/{config.stage}/trial_{config.model}/"

class Isofitts:
    def __init__(self, num_circles=8, num_trials=24, logging=True, fps=60, width=1920, height=1080):
        pygame.init()
        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode([width, height])
        self.clock = pygame.time.Clock()
        
        # logging information
        self.log_dictionary = {
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
        self.big_rad   = 400
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.done = False
        self.VEL = 30
        self.dwell_time = 1
        self.num_of_circles = num_circles 
        self.max_trial = num_trials
        self.width = width
        self.height = height
        self.fps = fps
        self.logging = logging
        self.trial = 0
        self.cursor_size = 14

        # interface objects
        self.circles = []
        self.cursor = pygame.Rect(self.width//2 - 7, self.height//2 - 7, self.cursor_size, self.cursor_size)
        self.goal_circle = -1
        self.get_new_goal_circle()
        self.current_direction = [0,0]

        self.p, self.odh, self.smi = setup_live_processing()
        smi_classifier = config.shared_memory_items
        for i in smi_classifier:
            i.append(Lock())
        self.classifier = setup_classifier(self.odh, save_location, smi_classifier)

        # Socket for reading EMG
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        self.sock.bind(('127.0.0.1', 12346))

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
        pygame.draw.circle(self.screen, self.YELLOW, (self.cursor.x + 7, self.cursor.y + 7), 7)

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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.current_direction[0] -= self.VEL
                elif event.key == pygame.K_RIGHT:
                    self.current_direction[0] += self.VEL
                elif event.key == pygame.K_UP:
                    self.current_direction[1] -= self.VEL
                elif event.key == pygame.K_DOWN:
                    self.current_direction[1] += self.VEL
            
        self.current_direction = [0,0]
        data, _ = self.sock.recvfrom(1024)
        data = str(data.decode("utf-8"))
        if data:
            dof_outputs = data.split(' ')[:2]
            dof_outputs = [float(i) for i in dof_outputs]
            self.current_direction[0] = dof_outputs[0] * self.VEL
            self.current_direction[1] = -1* dof_outputs[1] * self.VEL # pygame uses down as positive
            
            if self.logging:
                self.log(dof_outputs)

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

    def move(self):
        # Making sure its within the bounds of the screen
        if self.cursor.x + self.current_direction[0] > 0 + self.cursor_size//2 and self.cursor.x + self.current_direction[0] + self.cursor_size//2 < self.width:
            self.cursor.x += self.current_direction[0]
        if self.cursor.y + self.current_direction[1] > 0 + self.cursor_size//2 and self.cursor.y + self.current_direction[1] + self.cursor_size//2 < self.height:
            self.cursor.y += self.current_direction[1]
    
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

    def log(self, label):
        circle = self.circles[self.goal_circle]
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_circle'].append((circle.centerx, circle.centery, circle[2]))
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.centerx, self.cursor.centery, self.cursor[2]))
        self.log_dictionary['class_label'].append(label) 
        self.log_dictionary['current_direction'].append(self.current_direction)

    def save_log(self):
        file_parts = save_location.split("/")
        for f, fi in enumerate(file_parts[:-1]):
            if not os.path.exists("/".join(file_parts[:f+1])):
                os.mkdir("/".join(file_parts[:f+1]))
        with open(save_location + "output.pkl", 'wb') as f:
            pickle.dump(self.log_dictionary, f)

    def run(self):
        while not self.done:
            # updated frequently for graphics & gameplay
            self.update_game()
            pygame.display.update()
            self.clock.tick(self.fps)
        pygame.quit()

    def get_hardware_process(self):
        return self.p
    



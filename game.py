from multiprocessing import Lock, Process
import socket
import pickle
import time
from utils import setup_live_processing
from models import setup_classifier
from config import Config
config = Config()
import memoryManager
import adaptManager

# TODO: Modify this to wrap around IsoFitts instead of Unity game
# Can we just use the libemg IsoFitts, or do we need to make our own? Will depend on if appending to a dictionary is fast enough I guess...
# Also want to use shared memory instead of sockets I think
class Game:
    def __init__(self):
        self.unity_port, self.memory_port, self.classifier_port = 12347, 12348, 12349
        self.classifier_smi = config.shared_memory_items
        for item in self.classifier_smi:
            item.append(Lock())
        self.p, self.odh, smi = setup_live_processing()
        self.classifier = setup_classifier(self.odh, 
                                           f"data/subject{config.subjectID}/{config.stage}/trial_{config.model}/",
                                           self.classifier_smi)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', self.unity_port))
        self.sock.sendto(bytes(str("READY"), "utf-8"), ("localhost",12346)) # send ready to unity

    def run(self):
        
        started=False

        memoryProcess = Process(target = memoryManager.worker, daemon=True, 
                                args=
                                (
                                    self.classifier_port, self.unity_port, self.memory_port,
                                    f"data/subject{config.subjectID}/{config.stage}/trial_{config.model}/",
                                    config.negative_method,
                                    self.classifier_smi
                                )
        )
        print("--- Ready for Unity ---")
        while not started:
            try:
                received_data,_ = self.sock.recvfrom(1024)
                data = received_data.decode('utf-8')
                if data == "GAMEPLAY":
                    global_timer = time.perf_counter()
                    started = True
            except:
                pass
        # we no longer need this socket. the game has started.
        self.sock.close()

        memoryProcess.start()
        adaptManager.worker(self.memory_port, self.classifier_port, f"data/subject{config.subjectID}/{config.stage}/trial_{config.model}/",self.classifier)

        while (time.perf_counter() - global_timer < config.game_time):
            time.sleep(1)
                
        self.clean_up()
        # because we are running memory process with a daemon it dies when the main process dies
    
    def get_hardware_process(self):
        return self.p

    
    def clean_up(self):
        with open(f"data/subject{config.subjectID}/{config.stage}/trial_{config.model}/final_mdl.pkl",'wb') as handle:
            pickle.dump(self.classifier.classifier, handle)
        self.odh.stop_all()
        self.classifier.stop_running()
        self.p.signal.set()
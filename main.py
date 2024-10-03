from screen_guided_training import ScreenGuidedTraining, prepare_model_from_sgt #prepare_blank_model, visualize_sgt
from isofitts import Isofitts
import utils
from config import Config
config = Config()

def main():
    if config.stage == "sgt":
        # sgt = ScreenGuidedTraining()
        # sgt.run()
        prepare_model_from_sgt()
        # visualize_sgt()
        # prepare_blank_model()
        # p = sgt.get_hardware_process()
    elif config.stage == "fitts":
        fitts = Isofitts()
        fitts.run()
        p = fitts.get_hardware_process()
    utils.cleanup_hardware(p)


if __name__ == "__main__":
    main()
import sys
import os 
from GUI import GUI
from ManhuaTranslator import ManhuaTranslator

# Resources 
MODEL_PATH = os.getenv("SAM_PATH")
PROCESSOR_PATH =  os.getenv("SAM_PROCESSOR")
TRANSLATOR_NAME =  os.getenv("TRANSLATOR_NAME")

# Image's path
IMAGE_PATH =  os.getenv("SAM_PATH")
RESULT_PATH = os.getenv("SAM_PATH")

def main()->None:

    # 1. Load models 
    manhua_translator = ManhuaTranslator(MODEL_PATH,PROCESSOR_PATH,TRANSLATOR_NAME)

    # 2. Get text location input 
    gui = GUI()
    image_PIL , input_boxes ,input_points = gui.get_inputs(IMAGE_PATH)

    if image_PIL == None and input_boxes == None and input_points == None : # Exit if nothing was chosen 
        sys.exit(0)

    # 3. Extract text , inpaint on the image and save result 
    manhua_translator.manhua_translator(image_PIL,
                                        RESULT_PATH,
                                        input_boxes,
                                        input_points)
    

if __name__ == "__main__":
    main()

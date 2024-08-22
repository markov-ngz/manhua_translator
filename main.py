import sys
import os 
from GUI import GUI
from ManhuaTranslator import ManhuaTranslator

MODEL_PATH = "../models/sam_model/"
PROCESSOR_PATH = "../models/sam_processor/"
TRANSLATOR_NAME = "Helsinki-NLP/opus-mt-ko-en"



def main()->None:
    
    path = "image.png"
    save_path = "result.png"
    
    manhua_translator = ManhuaTranslator(MODEL_PATH,PROCESSOR_PATH,TRANSLATOR_NAME)
    gui = GUI()
    image_PIL , input_boxes ,input_points = gui.get_inputs(path)

    if image_PIL == None and input_boxes == None and input_points == None : 
        sys.exit(0)

    manhua_translator.manhua_translator(image_PIL,
                                        save_path,
                                        input_boxes,
                                        input_points)
    

if __name__ == "__main__":
    main()
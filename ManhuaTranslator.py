from manhua_translator.KoreanExtractor import KoreanExtractor
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
import math

class ManhuaTranslator(KoreanExtractor):

    def __init__(self, sam_model: str, sam_processor: str, translator_name: str, device: str = "cpu") -> None:
        super().__init__(sam_model, sam_processor, translator_name, device)

    def manhua_translator(self,image_PIL:Image,save_path:str, input_boxes:list, input_point:np.array,):
        # Extract the texts 
        texts = self.extract_korean(image_PIL,input_boxes,input_point)

        image_inpainted = self.write_all_texts(texts,image_PIL)

        image_inpainted.save(save_path)

    def draw_text_within_boundary(self,image_original:Image, text:str, boundary_width:int, font_color:str="black",police:str="arial.ttf"):
        """
        From an image , 
            a text, 
            the original font_size ,
            the max width, 
            the  police
        Return an image with the text written 
        """
        

        font_size = 100 
        
        while font_size > 0 :

            image = Image.fromarray(np.copy(np.array(image_original))).convert("RGB")
            # PIL.Image -> PIL.ImageDraw
            draw = ImageDraw.Draw(image)
            
            # font 
            font = ImageFont.truetype(police, font_size)

            # Break the text into lines to fit within the boundary
            lines = []
            current_line = ""

            for word in text.split():
                # add the word to the current line 
                test_line = current_line + " " + word
                # get the width of the text 
                text_width = draw.textlength(test_line, font)
                
                # if the width does not takes too much space
                if text_width <= boundary_width:
                    # validate the line
                    current_line = test_line
                else:
                    # if it is out of boundary, stop the line writing
                    lines.append(current_line.strip())
                    current_line = word

            # add the last line
            lines.append(current_line.strip())

            # writing text
            y_position = 0  # line height position 
            for line in lines:
                draw.text((0, y_position), line, font=font, fill=font_color)
                y_position += font_size  # adjust as needed
            if y_position < image.size[1] : 
                return image# if the text does not go overboard in height

            font_size -=1

    def write_text(self,texts:dict,idx:int, image:Image,police:str="arial.ttf") -> Image.Image :
        """
        From : 
        a dict with keys : ["text","coordinates","font_size"]
        a PIL.Image RGB
        Return the Image with the "text" written within "coordinates" of the "font_size"
        """
        #---A. SETUP------------------------------------------------------------------------------------------
        # A.1. Unstack bounding box coordinates
        y_min, y_max , x_min, x_max = texts[idx]["coordinates"]

        # A.2. Copy original_image
        image_cleaned = np.array(image)

        # A.3. Background color 
        background_color = image_cleaned[y_min-1][x_min] 

        # A.4. Rectangle where the text will be written within
        text_array = np.ones((y_max-y_min,x_max-x_min,3))*background_color
        # np.array ((height, width,3), float64) -> np.array ((height, width,3), uint8) -> PIL.Image RGB
        image_textual = Image.fromarray(text_array.astype("uint8"),"RGB")

        # A.5. Font Size
        # get the font size used and diminish it by 10 % for the text to fit well 
        font_size = texts[idx]["font_size"]  - math.ceil(texts[idx]["font_size"]*0.1)

        # A.6. Font color 
        # if the background is dark , write the text in white 
        font_color = "black"
        if background_color.max() < 101 : 
            font_color = "white"
        # A.7. Max width of the image
        boundary_width = text_array.shape[1]

        #---B. WRITING-----------------------------------------------------------------------------------------

        # Image with drawn text
        drawn_text = self.draw_text_within_boundary(image_textual, texts[idx]["text"], boundary_width,font_color,police=police)

        #---C. REPLACING------------------------------------------------------------------------------------------
        # PIL.Image BW -> PIL.Image RGB -> np.array (height,width,3)
        np_drawn_text_rgb = np.array(drawn_text.convert("RGB"))

        # Replace the original text by the translated one 
        image_cleaned[y_min:y_max , x_min : x_max] = np_drawn_text_rgb

        # np.array ((height, width,3), uint8) -> PIL.Image RGB
        new_image= Image.fromarray(image_cleaned,"RGB")

        return new_image
    
    def write_all_texts(self,texts:dict,image:Image,police="arial.ttf")->Image.Image:
        """
        From a dict with integer keys { 0: {"texts":,"coordinates":,"font_size": }}, the original image 
        Return  an image with all the written text instead of the original ones
        """
        new_image = image
        i=0

        while i < len(texts):
            new_image = self.write_text(texts,i, new_image,police=police)
            i+=1
        return new_image
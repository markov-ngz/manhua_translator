from transformers import SamModel, SamProcessor, pipeline
import pytesseract
from PIL import Image
from torch import no_grad, Tensor
import numpy as np 
import math 
import cv2

class ComicExtractor():


    def __init__(self,sam_model:str,sam_processor:str,translator_name:str,lang:str,device:str = "cpu") -> None:
        """
        
        """

        self.load_model(sam_model,sam_processor,translator_name)
        self.device = device
        self.lang = lang

    def load_model(self,sam_model:str,sam_processor:str,translator_name:str)->None:
        """Load Model from local path"""
        # load Segment Anything Model from HD 
        self.sam = SamModel.from_pretrained(sam_model,local_files_only=True)
        self.sam_processor = SamProcessor.from_pretrained(sam_processor,local_files_only=True)
        self.translator = pipeline("translation", model=translator_name)

    def extract_text(self, image_PIL:Image,input_boxes:list, input_point:np.array)->dict:
        """
        From positions of text in the image, the image and the models
        Extract the text selected and translate it 
        """

        image_embeddings = self.embed(image_PIL)

        texts = {}

        for i in range(len(input_boxes[0])):
            # predict the mask and scores
            masks, _ = self.segment(image_PIL,
                                    image_embeddings,
                                    [[input_boxes[0][i]]],
                                    np.array([input_point[i]]))
            # get it 
            mask_image, binary_mask = self.get_mask(masks,0)
            # mask the img
            masked_image = self.apply_mask(image_PIL,mask_image, binary_mask)
            # crop it 
            cropImg, coordinates = self.crop(image_PIL,masked_image)
            # pad 
            padded_crop_img = self.paddings([cropImg],[cropImg.shape], 0.1)
            # split between lines
            images, images_shape  = self.split_lines(padded_crop_img[0], min_space_between_words = 1)
            # pad again 
            padded_images  = self.paddings(images,images_shape, 0.1 )
            #extract the text 
            line_text = self.predict_line(padded_images, config=r"--psm 11 --oem 3", confidence_score=40, lang=self.lang)

            texts[i] = {}
            texts[i]["text"] = self.translate(line_text) # translated text
            texts[i]["coordinates"] = coordinates # coordinates to overwrite the text
            texts[i]["font_size"] = sum([shape[0] for shape in images_shape]) // len([shape[0] for shape in images_shape])# font size used

        return texts


    def embed(self,image:Image)->Tensor:
        """
        Get the image's embedding
        Return the embeddings of the image ( torch.Tensor ) 
        """
        inputs = self.sam_processor(image, return_tensors="pt").to(self.device)
        image_embeddings = self.sam.get_image_embeddings(inputs["pixel_values"])
        
        return image_embeddings   
    

    def segment(self,image:Image, image_embeddings, input_boxes:list, input_point:list)->Tuple:
        """
        Segment the image 
        Return : 
        - masks : segmentation masks ( torch.Tensor ) 
        - scores : confidence scores ( list ) 
        """
        # preprocess the image with boxes 
        inputs = self.sam_processor(image, input_boxes=input_boxes,input_point=input_point, return_tensors="pt").to(self.device)

        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        # predict
        with no_grad():
            outputs = self.sam(**inputs)

        # masks[0] are masks see below to get one mask as 2D array 
        masks = self.sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        # confidence scores
        scores = outputs.iou_scores

        return masks, scores
    
    def get_mask(self,masks,idx:int)-> tuple :
        """
        Return a tuple of torch.Tensor : 
        (height,width,1) & (height,width) 
        The first tensor has just unsqueezed one dim 
        """

        # squeeze the batch size
        masks_squeezed = masks[0].squeeze()

        # First mask, float():bool-> 1,0 
        binary_mask =masks_squeezed[idx].float()

        # Height and Weight
        h, w = binary_mask.shape[-2:]

        # Add a third dim 
        mask_image = binary_mask.reshape(h, w, 1)

        return mask_image, binary_mask
    
    def apply_mask(self, image:Image,mask_image:Tensor, binary_mask:Tensor)->np.array:
        """
        From 2 masks ( mask_image is unsqueezed while the other is not )
        Return an np.array of the image computed by the mask 
        """
        # (height, width, 1) -> (height * width,3) -> unstack : height * width -> zip -> list
        nonzero_coords = list(zip(*mask_image.nonzero()))

        # get coordinates 
        x_min = min(coord[0] for coord in nonzero_coords)
        y_min = min(coord[1] for coord in nonzero_coords)
        x_max = max(coord[0] for coord in nonzero_coords)
        y_max = max(coord[1] for coord in nonzero_coords)

        # Convert coordinates to integers
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # multiply the image by the mask 
        masked_image = np.array(image.convert("L"))*np.array(binary_mask)

        return masked_image
    
    def crop(self,image:Image,masked_image:np.array)->tuple:
        """
        From a np.array of a masked image where only the wanted part is seen,
        Return a np.array of a cropped image to only get the wanted part
        and a tuple of the coordinates ( y min, y max , x min, x max)
        """
        prediction = np.squeeze(masked_image).astype(np.float32)
        threshold = 0.2
        xmin,ymin,xmax,ymax = (-1,-1,-1,-1)
        for j in range(image.size[1]):
            for i in range(image.size[0]):
                if prediction[j,i] <= threshold: continue
                xmin = i if xmin == -1 or xmin > i else xmin
                xmax = i if xmax == -1 or xmax < i else xmax
                ymin = j if ymin == -1 or ymin > j else ymin
                ymax = j if ymax == -1 or ymax < j else ymax
        cropImg = np.array(image.convert("L"))[ymin:ymax,xmin:xmax]

        coordinates = (ymin,ymax,xmin,xmax)

        return cropImg, coordinates
    
    def is_it_the_end(self,i:int,min_space_between_words:int,width:int,std_width:np.array)-> bool :
        """
        From : 
            the width index i, 
            the minimum space between words wanted, 
            the width of the image,
            the std's distribution
        Return :
            Boolean if the word is considered finish or not
        """
        # it is the end of words
        end = True
        # looks at min_space_between_words ahead 
        for j in range(i,i+min_space_between_words):
            # if j out of range of the image width ( for the end of the picture )
            if j >= width :
                return end
            # if there is perturbation ( meaning that there is a letter )
            if std_width[j] > 3:
                # that would mean that this not the end 
                end = False
                return end
        
        return end
    
    def split_lines(self,text_img:np.array, min_space_between_words:int = 5)->list:
        """
        From a one line text image,
        Return a List of np.array 
        Each word is split when it is too far apart from his neighbor by 
        """

        # Transpose to iter over the width instead of height
        # cropped_img_T = np.transpose(text_img)

        # Std's distribution
        std_pixels =[row.std() for row in text_img]
        
        # list -> np.array
        std_pixels = np.array(std_pixels)

        # Initialisation
        height = std_pixels.shape[0]
        words_x = []
        i = 0

        on_word = False # at the beginning no words are currently read 
        block_beginning = False 

        for h in std_pixels:  # Look at the std over by the width
            
            if h > 1 : # not uniform distrib => there is a character

                if  on_word == False : # a word reading has not begun 
                        block_beginning = i
                        on_word = True # => initialize one

            else : # uniform distrib h == 0
                

                if on_word :  # a word reading is currently on 

                    # any character near it ? 
                    end = self.is_it_the_end(i,min_space_between_words,height,std_pixels)

                    if end : # indeed no character after the current end of this word

                        # index of the end of the word
                        block_ending = i - 1
                        if block_ending - block_beginning >= min_space_between_words :
                            # Add coordinates of it 
                            words_x.append((block_beginning,block_ending))
                        # set : no words are currently read
                        on_word = False
            i+=1 
        
        # for this line get all the block of "words"
        line_words = [text_img[x:z,0:text_img.shape[1]] for x,z in words_x]

        

        # for those block of "words", get the shape
        line_words_shape = [i.shape for i in line_words]


        return line_words, line_words_shape
    
    def paddings(self,line_words:list, line_words_shape:list, pad_proportion:float = 0.1)->list:
        """
        From list of multiple Words, a list of their shape(tuple required), and a given_proportion 
        Pads them by a given proportion in order to center them and have a more precise prediction
        Return a list of np.array images padded 
        """
        # Values to pad 
        pads = [math.ceil(i.shape[1] * pad_proportion) for i in line_words]

        #----1. Width Padding ------------------------------------------------------------------------------------

        i = 0
        padded_width_img = []

        while i < len(line_words):
            # White rectangle 
            pad_width = np.ones((line_words_shape[i][0],pads[i]))*255
            # Add left and right the white rectangle 
            padded_value = np.concatenate([pad_width,line_words[i],pad_width],axis=1)
            # Add it to the list
            padded_width_img.append(padded_value)
            i+=1

        #----2. Height Padding ---------------------------------------------------------------------------------------

        j= 0 
        padded_imgs = []

        while j < len(padded_width_img):
            pad_height = np.ones((pads[j], padded_width_img[j].shape[1]))*255
            padded_value = np.concatenate([pad_height,padded_width_img[j],pad_height],axis=0)
            padded_imgs.append(padded_value)
            j+=1

        # return the list of it 
        return padded_imgs
    
    def guess_word(self,word_image:np.array, my_config:str = r"--psm 11 --oem 1", confidence_score:int = 70, lang="eng")->list:
        """
        From an image preprocessed , a given config 
        Return a list of the texts predicted
        """

        # otsu's function need the data type of the element to be uint8
        word_image_uint8 = word_image.astype("uint8")
        
        
        (thresh, im_bw) = cv2.threshold(word_image_uint8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        data = pytesseract.image_to_data(im_bw,output_type=pytesseract.Output.DICT, config=my_config, lang=lang)

        text = []
        for i in range(len(data['text'])):
            if float(data['conf'][i]) > confidence_score : 
                text.append(data['text'][i])
                
        return text
    
    def predict_line(self,line_words_imgs:list,config: str = r"--psm 7 --oem 1", confidence_score:int = 70, lang="eng")->str:
        """
        From a list of np.array's word's image and a given config
        Return a str of the text of image which should have the text over a single line 
        """
        
        vocab = []

        for pic in line_words_imgs:
            predict =self.guess_word(pic,  my_config=config, confidence_score=confidence_score, lang=lang)
            vocab.append(' '.join(predict))
            
        vocab = ' '.join(vocab)

        return vocab
    def translate(self,text:str)->str:
        translation = self.translator(text)
        return translation[0]["translation_text"]

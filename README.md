# manhua_translator

Repository dedicated to the development of an automated mahnua translator.

## 1. User Guide

1- Install dependencies and run the script ```main.py```  <br>   <br>
Your image will open up and you will be asked to draw a rectangle around your text : <br>
2- Make 2 clicks ( left-high corner and bottom-right ) then press key 'q'  <br>  <br>
A new image shall appear and your rectangle shall be drawn, now add the point input : <br>
3- Make one click on the text , then press key 'q'  <br>  <br>
The point shall appear :  <br>
4- Press again 'q'  <br>  <br>
The image shall close and you will be asked for confirmation through terminal output :  <br>
5- Type 'yes' to continue  <br>  <br>
Your image will then be processed.  <br>
Your can retrieve your result to the path set by ```RESULT_PATH``` variable. 

## 2. Environment Used 

<b> Python version </b> : 3.11.4  <br>
Set the following environment variables to match your configuration.
```
# Path of the image to translate
IMAGE_PATH = 

# Path where the result image will be stored
RESULT_PATH = 

# SAM's model & processor directory path 
SAM_PATH = 
SAM_PROCESSOR = 

# Translation model to use the pipeline ( value "Helsinki-NLP/opus-mt-ko-en" ) 
TRANSLATOR_NAME =
```

<hr>

## 3. Example 


##  4. Resource's reference : 

### Segmentation Anything model : 
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
- huggingface : https://huggingface.co/facebook/sam-vit-base 
- paper : https://arxiv.org/pdf/2304.02643v1 

### Translation model :
```
@inproceedings{tiedemann-2020-tatoeba,
    title = "The {T}atoeba {T}ranslation {C}hallenge {--} {R}ealistic Data Sets for Low Resource and Multilingual {MT}",
    author = {Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the Fifth Conference on Machine Translation",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wmt-1.139",
    pages = "1174--1182"
}
```

hugging face : https://huggingface.co/Helsinki-NLP/opus-mt-ko-en

### Tesseract-OCR :
- repository : https://github.com/tesseract-ocr/tesseract 
- license : https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE 
!! : in order to extract korean data, please make sure you have the language installed 

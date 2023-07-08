# Advanced Automatic Number Plate Recognition System [ANPR]
This project focuses on developing a robust number plate recognition system capable of
detecting and recognizing "fancy" number plates, which include number plates with different font
sizes and languages in a single image. The project components are designed to work together as
an integrated system, encompassing both number plate detection and recognition.
In the first phase, a custom dataset is curated, consisting of approximately 3000 images collected
from various sources, and manually labeled. 

Our Grand Collection of Dataset- https://drive.google.com/drive/folders/195oOIbT3DTY7M3rYSPOF-i6mzdiqGavD?usp=sharing

The YOLOv8 object detection model is trained on this dataset using Google Colab. The model is 
specifically trained to handle the nuances of fancy number plates, including variations in font 
sizes and languages. Preprocessing techniques, such asgrayscale scaling and image contouring, 
are employed with the aid of OpenCV to improve the detection accuracy. After 100 epochs of training,
the model achieves an impressive accuracy of 80% and is validated using separate training and testing 
datasets. The trained model is thenemployed to detect number plates in video frames from sources 
such as YouTube videos, webcam feeds, and CCTV footage.

The second phase of the project focuses on number plate recognition. The integration of EasyOCR
combined with OpenCV, allows for accurate text extraction from the detected number plates.
Preprocessing steps, including grayscale conversion, are implemented to optimize the text
extraction process. Moreover, additional libraries such as Tesseract, Pytesseract, and OpenALPR
are utilized to further refine the recognition process and enhance the accuracy of character
extraction from the fancy number plates. The recognized text, accompanied by confidence scores,
is displayed, and bounding boxes are overlaid on the corresponding images.

This project offers a comprehensive solution for detecting and recognizing fancy number
plates with varying font sizes and languages in a single image. The seamless integration of
YOLOv8 for detection and EasyOCR, along with other supporting libraries, demonstrates an
effective approach to handle the intricacies associated with such challenging scenarios. 


Sample Output of our Developed Model -
<img src="https://drive.google.com/file/d/1c_ZaCO2ChQprrlaI20pXjjGZJ7NCpEYa/view?usp=sharing)https://drive.google.com/file/d/1c_ZaCO2ChQprrlaI20pXjjGZJ7NCpEYa/view?usp=sharing" alt="Number plate recognition">
![7](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/9fd100cc-cd6c-434e-b778-382f24cc22d8)

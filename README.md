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

1] Number Plate Detection -

![10](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/316583a5-7d81-4f36-b7d4-adc2b5562fa4)
![4](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/3e046f30-2063-4e34-988c-e6b8426e7a69)

Normal Number plate Detection and Recogniton-

![5](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/7c873989-1f8b-486e-82e7-95e0a4bce30c)
![6](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/2626a7ec-4485-418f-97c4-62ba2083f988)

2] Fancy Number Plate (Plates with different fonts,designs,styles and languages)

![7](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/9fd100cc-cd6c-434e-b778-382f24cc22d8)
![8](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/f304415e-4705-4d0e-8722-b44757bec80c)
![9](https://github.com/samay-jain/Advanced-Automatic-Number-Plate-Recognition-System-ANPR-/assets/116068471/2d24b807-08fb-4282-ad75-87c10f9af9a2)


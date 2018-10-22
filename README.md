# form-extractor-ocr

Template based form extractor OCR. 
Extract handwritten text from bank form scanned image (any form scanned copy), using template matching, indivicual box extraction and OCR.
Train your own character and alphabet OCR with pytesseract.
</br>

#### Input scanned form snippet-
<img src="https://user-images.githubusercontent.com/12294956/47312583-697cfe00-d65a-11e8-930a-e15fd67a5bb1.png">

#### Tagged output snippet-
Prediction output with red tagging shows a confidence score < 80% and blue tagging shows a confidence >= 80%.

<img src="https://user-images.githubusercontent.com/12294956/47312584-697cfe00-d65a-11e8-95f7-5554a04bb0b8.png">

</br>

<b>NOTE</b>: Current model was trained on 10 images of each handwritten characters from (a-z) & (A-Z) and 10 images of each handwritten numbers from (0-9), that's why the prediction accuracy is poor, but with more data the prediction can be improved.
Use handwriten character or number images like data/crop_image/* images for training process.

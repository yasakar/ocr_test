import pandas as pd
import os
import shutil
import glob
import cv2
from ultralytics import YOLO
import easyocr
import PIL.ExifTags as ExifTags
from PIL import Image
import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

numDF = pd.DataFrame(columns=['ImageName', 'num'])
img_folder_path = "./TargetFolder/"
img_file_list = glob.glob(os.path.join(img_folder_path,"*.jpg"))

model = YOLO(task="segment")
model = YOLO(model="yolov8n-seg.pt")
model = YOLO(model="best.pt")
# reader = easyocr.Reader(['en'])
endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
subscription_key = os.environ['COMPUTER_VISION_KEY']
text_recognition_url = endpoint + "computervision/imageanalysis:analyze?features=caption,read&model-version=latest&language=en&api-version=2023-02-01-preview"

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/octet-stream'
}

if not os.path.isdir("output"):
    os.makedirs("output")
if not os.path.isdir("renamedImages"):
    os.makedirs("renamedImages")

#CSVとそのファイル名を作成するためのリスト
detectNumList = []
detectStringList = []
datetimeList =[]

for img_path in img_file_list:
    img = cv2.imread(img_path)
    img_PIL = Image.open(img_path)

    # OCR_results = reader.readtext(img)  # easyocr
    # ******* azure_readAPI　********
    with open(img_path, 'rb') as image_file:
        image_data = image_file.read()

    response = requests.post(
    url=text_recognition_url,
    headers=headers,
    data=image_data
    )
    response.raise_for_status()
    response = response.json()
    OCR_results = response["readResult"]["content"]
    OCR_results = OCR_results.replace('\n', '')
    # ******* azure_readAPI　********

    # ******* YOLO v8 ********
    ObjectDetection_results = model.predict(img,
                                             project="runs", #出力先
                                             name="predict", #ディレクトリ名
                                             exist_ok=False, #上書き
                                             conf=0.5,
                                             save=True
                                             )
    # ******* YOLO v8 ********
    
    detectString = ""
    #OCRで検出された文字抽出、リストに格納
    for result in OCR_results:
        n = result
        detectString += n
    detectStringList.append(detectString)
    print("detectString",detectString)
    
    #きゅうりの本数をリストに格納
    for result in ObjectDetection_results:
        detectNumList.append(len(result))
        
    exif_data = img_PIL._getexif()
    if exif_data is not None:
        exif_dict = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
        if "DateTimeOriginal" in exif_dict:
            #撮影日時に基づく新規ファイル名を準備
            file_dateTime = datetime.datetime.strptime(exif_dict["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            file_dateTime = file_dateTime.strftime("%Y-%m-%d_%H-%M-%S")
            datetimeList.append(file_dateTime)
        else:
            #DateTimeOriginalが存在しなかった場合
            datetimeList.append("2023-0-0")
    else:
        datetimeList.append("2023-0-0")
    img_PIL.close()

print("OCRで検出された文字列の数",len(detectStringList))
print("元画像のメタデータ（datetime）の数",len(datetimeList))

# 新しいファイル名を生成してリストに追加
ChangeFileNameList = [f"{detect}_{date}.jpg" for detect, date in zip(detectStringList, datetimeList)]
# 元画像を残したまま新しいディレクトリに名前変更済みの画像を格納
for index, img in enumerate(img_file_list):
    # 新しいファイル名を作成
    new_filename = ChangeFileNameList[index]
    # 元のファイルパス
    source_item = img
    # 新しいファイルパス
    destination_item = os.path.join("./renamedImages", new_filename)
    # ファイルをコピー
    shutil.copy2(source_item, destination_item)
  
#元画像を残したまま新しいディレクトリに名前変更済みの画像を格納
# contents = os.listdir(img_folder_path)
# for item in contents:
#     source_item = os.path.join(img_folder_path, item)
#     destination_item = os.path.join("./renamedImages", item)
#     shutil.copy2(source_item, destination_item)
# for index,img in enumerate(img_file_list):
#     os.rename(img,"./renamedImages/" + detectStringList[index] + "_" + datetimeList[index] + ".jpg")
    

#結果をCSVに出力
result_list = [[item1, item2] for item1, item2 in zip(ChangeFileNameList, detectNumList)]
numDF = pd.DataFrame(result_list, columns=numDF.columns)
numDF.to_csv('output/cucumber_num.csv', index=False)
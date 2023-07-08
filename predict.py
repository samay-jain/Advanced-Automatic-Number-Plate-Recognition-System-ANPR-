
import hydra
import torch
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import easyocr
#import pytesseract

#TESSDATA_PREFIX = 'C:/Program Files (x86)/Tesseract-OCR/tessdata'
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
#flg=False
#reader = easyocr.Reader(['en'],gpu=True)
#reader = easyocr.Reader(['en', 'hi', 'mr'], gpu=True)
def ocr_image(img,coordinates):
    x,y,w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])
    img = img[y:h,x:w]

    #print('In the OCR_Image function')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = reader.readtext(gray)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
    
    return str(text)

reader = easyocr.Reader(['en'],gpu=True)
def tesseract_recognition(img,coordinates):
    
    
    x,y,w,h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])
    img = img[y:h,x:w]
    #text_data = pytesseract.image_to_string(img, lang='eng')
    #print(text_data)
    print('In the Tessareract Recognition')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    text=""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
    
    return str(text)
    #return text_data


class DetectionPredictor(BasePredictor):
    count=0
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,self.args.conf,self.args.iou,agnostic=self.args.agnostic_nms,max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
 
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt: 
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh) 
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                #label = None if self.args.hide_labels else (
                    #self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                label = 'Number Plate'
                text_ocr = ocr_image(im0,xyxy)
                label = text_ocr
                #label = tesseract_recognition(im0,xyxy)
                self.annotator.box_label(xyxy, label, color=colors(c, True))

            #if self.args.save_crop:
            imc = im0.copy()
            save_path = self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg'
            save_one_box(xyxy, imc, file=save_path, BGR=True)

            #Convert cropped image to grayscale
            #cropped_img = cv2.imread(str(save_path))
            #gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            #Save grayscale image
            #gray_save_path = str(save_path).replace('.jpg', '_gray.jpg')
            #cv2.imwrite(gray_save_path, gray_img)
            #cv2.imwrite(str(save_path), gray_img)
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    
    if not hasattr(cfg, 'show'):
        cfg.show = True

    cfg.show = True
    predictor = DetectionPredictor(cfg)
    print(cfg.source)
    predictor()

if __name__ == "__main__":
    predict()

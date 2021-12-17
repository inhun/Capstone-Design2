import numpy as np
import cv2
import time
import threading
import requests
import logging

from video_capture import *
from models.model import *
from PIL import Image
import torchvision.transforms as transforms

from utils.datasets import *
import torch
from utils.utils import *
from utils.openmax import *


class MLEngine:
    def __init__(self):
        self.init_logger()

        self.cap = cv2.VideoCapture('data/videos/sw/7.mp4')
        
        # self.cap = cv2.VideoCapture('rtsp://icns:iloveicns@icns2.iptimecam.com:21128/stream_ch00_0')
        
        self.bs = cv2.createBackgroundSubtractorMOG2()

        self.model = HunNet().to(torch.device('cuda'))
        self.model.load_state_dict(torch.load('checkpoints/background4/40.pth'))
        self.model.eval()

        self.A0_list = self.loadLogitVector('LogitVector/background/class2/0_LogitVector_Average.txt')
        self.A1_list = self.loadLogitVector('LogitVector/background/class2/1_LogitVector_Average.txt')

        self.n = nn.Softmax(dim=0)

        self.fire_count = 0
        self.smoke_count = 0

        self.init_time = time.time()

        self.thr = threading.Thread(target=self.danger)
        self.thr.daemon = True
        self.thr.start()

        self.isAbort = False
        self.draw_frame = 0
        self.danger_frame = 0


    def init_logger(self):
        logger = logging.getLogger('Main.MLEngine')
        logger.setLevel(logging.INFO)
        self.logger = logger


    def loadLogitVector(self, path):
        with open(f'{path}', 'r') as f:
            A1 = f.readline()
            return A1.split(' ')


    def danger(self):
        cv2.namedWindow('Fire Smoke Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fire Smoke Recognition', 1280, 720)

        while self.cap.isOpened():
            isSmoke = False
            isFire = False

            _, frame = self.cap.read()

            fgmask = self.bs.apply(frame)
            back = self.bs.getBackgroundImage()
            cnt, _, stats, _ = cv2.connectedComponentsWithStats(fgmask)

            draw_frame = frame.copy()
            danger_frame = frame.copy()

            for i in range(1, cnt):
                dst = crop_box(frame, stats, i)
                x, y, w, h, s = stats[i]

                if s < 500:
                    continue

                if w/h > 5:
                    continue
                if h/w > 5:
                    continue
                    
                if x+w >= 1920:
                    if y + h >= 1080:
                        dst = frame[y:1080, x:1920].copy()
                    else:
                        dst = frame[y:y+h, x:1920].copy()
                else:
                    if y + h >= 1080:
                        dst = frame[y:1080, x:x+w].copy()
                    else:
                        dst = frame[y:y+h, x:x+w].copy()


                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                dst = transforms.ToTensor()(dst)
                dst = resize(dst, 256).unsqueeze(0).to(torch.device('cuda'))
                

                with torch.no_grad():
                    outputs = self.model(dst)
                
                # print(outputs)

                if outputs[0][0] > 0.7:
                    cv2.rectangle(draw_frame, (x, y, w, h), (0 ,0, 255), 2)   
                    self.logger.info('smoke')
                    isSmoke = True
            
                elif outputs[0][1] > 0.5:
                    cv2.rectangle(draw_frame, (x, y, w, h), (255, 0, 0), 2)
                    self.logger.info('fire')
                    isFire = True
                

            cv2.imshow('Fire Smoke Recognition', draw_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            self.draw_frame = draw_frame

            cv2.rectangle(danger_frame, (1,1,1800, 960), (0,0,255), 2)
            cv2.putText(danger_frame, "Danger", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            self.danger_frame = danger_frame


            if isFire:
                self.fire_count += 1
            else:
                if self.fire_count == 0:
                    pass
                elif self.fire_count > 0:
                    self.fire_count -= 1

            if isSmoke:
                self.smoke_count += 1
            else:
                if self.smoke_count == 0:
                    pass
                elif self.smoke_count > 0:
                    self.smoke_count -= 1


            print(f'smoke: {self.smoke_count}')
            if self.isAbort == False and self.smoke_count > 5:
                print('smokesmokesmokesmokesmokesmokesmokesmokesmokesmokesmokesmokesmokesmoke')
                try:
                    res = requests.post('http://163.180.117.38:8281/api/accident/smoke/1')
                    res = requests.post('http://163.180.117.40:3000/api/SocketDataReceive', data={"websocketURL": "ws://163.180.117.39:10000", "posName": "ICNS11"})
                except:
                    pass
                self.isAbort = True

            if self.isAbort == False and self.fire_count > 5:

                try:
                    res = requests.post('http://163.180.117.38:8281/api/accident/fire/1')
                    res = requests.post('http://163.180.117.40:3000/api/SocketDataReceive', data={"websocketURL": "ws://163.180.117.39:10000", "posName": "ICNS11"})
                except:
                    pass
                self.isAbort = True

        self.cap.release()
        cv2.destroyAllWindows()



    def predict(self):

        cv2.namedWindow('Fire Smoke Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fire Smoke Recognition', 1280, 720)

        while self.cap.isOpened():
            isSmoke = False
            isFire = False

            _, frame = self.cap.read()

            fgmask = self.bs.apply(frame)
            back = self.bs.getBackgroundImage()
            cnt, _, stats, _ = cv2.connectedComponentsWithStats(fgmask)

            draw_frame = frame.copy()

            for i in range(1, cnt):
                dst = crop_box(frame, stats, i)
                x, y, w, h, s = stats[i]

                if s < 500:
                    continue
                # if s > 5000:
                #     continue

                if w/h > 5:
                    continue
                if h/w > 5:
                    continue
                    
                if x+w >= 1920:
                    if y + h >= 1080:
                        dst = frame[y:1080, x:1920].copy()
                    else:
                        dst = frame[y:y+h, x:1920].copy()
                else:
                    if y + h >= 1080:
                        dst = frame[y:1080, x:x+w].copy()
                    else:
                        dst = frame[y:y+h, x:x+w].copy()

                # cv2.rectangle(draw_frame, (x, y, w, h), (0, 255, 0), 2)

                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                dst = transforms.ToTensor()(dst)
                dst = resize(dst, 256).unsqueeze(0).to(torch.device('cuda'))
                

                with torch.no_grad():
                    outputs = self.model(dst)

                if outputs[0][0] > 0.7:
                    cv2.rectangle(draw_frame, (x, y, w, h), (0 ,0, 255), 2)    
                    self.logger.info('smoke')

                elif outputs[0][1] > 0.5:
                    cv2.rectangle(draw_frame, (x, y, w, h), (255, 0, 0), 2)
                    self.logger.info('fire')
                
                # result_score = self.n(outputs[0])
                # print(result_score)

            
                '''
                output_list = np.array(outputs.tolist()[0])
                scores = get_cdf(self.A0_list, self.A1_list, output_list)

                new_output = []
                unknown_score = 0
                outputs2 = outputs.squeeze(0).clone().detach()

                for output, score in zip(outputs2, scores):
                    new_output.append(output - (output*score))
                    unknown_score += output*score
                
                new_output.append(unknown_score)
                new_output = torch.stack(new_output)
                
                        
                result_score = self.n(new_output)

                print(result_score)
                if result_score[0] > 0.7:
                    isSmoke = True
                    pass
                # cv2.rectangle(draw_frame, (x, y, w, h), (0 ,0, 255), 2)
                # print('0')
                elif result_score[1] > 0.35:
                    cv2.rectangle(draw_frame, (x, y, w, h), (0, 0, 255), 2)
                    isFire = True
                    pass
                # cv2.rectangle(draw_frame, (x, y, w, h), (255, 0, 0), 2)
                # print('1')
                else:
                    cv2.rectangle(draw_frame, (x, y, w, h), (255, 0, 0), 2)
                    # print('unknown')
                '''
            cv2.imshow('Fire Smoke Recognition', draw_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            self.draw_frame = draw_frame

        self.cap.release()
        cv2.destroyAllWindows()

            
from ml_engine import *
import cv2
import websockets
import asyncio
import requests


me = MLEngine()
async def accept(websocket, path):
    while True:
        try:
            frame = me.danger_frame
            encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
            _ ,imgencode = cv2.imencode('.jpg', frame, encode_param)
            imgencode = np.array(imgencode)

            await websocket.send(imgencode.tobytes())
        except websockets.exceptions.ConnectionClosed as exc:
            print(exc.code)
            print("Client deisconnected")
            print("video capture release")


if __name__ == '__main__':
    
  
    start_server = websockets.serve(accept, '', 10000)

        
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()





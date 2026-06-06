import cv2
import numpy as np
from ultralytics import YOLO

class OtoparkAnalizSistemi:
    def __init__(self, model_path, video_path, output_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        
        # 16 Park yeri için parametreler
        self.x_baslangic = 50    
        self.y_ust = 100        
        self.y_alt = 300        
        self.kutu_genislik = 142 
        self.kutu_yukseklik = 200
        
        self.park_alanlari = []
        self._alanlari_olustur()

    def _alanlari_olustur(self):
        # Eğimleri buradan ayarla:
        egim_ust = 10    # Üst sıranın eğimi
        egim_alt = 12     # Alt sıranın eğimi
        ust_offset = 38  # Üst sırayı ayrıca sağa itmek için
        
        for i in range(8):
            # Üst sıra (Eğimli + Offsetli)
            x_ust = self.x_baslangic + (i * self.kutu_genislik) + (i * egim_ust) + ust_offset
            self.park_alanlari.append([x_ust, self.y_ust, x_ust + self.kutu_genislik, self.y_ust + self.kutu_yukseklik])
            
            # Alt sıra (Eğimli)
            x_alt = self.x_baslangic + (i * self.kutu_genislik) + (i * egim_alt)
            self.park_alanlari.append([x_alt, self.y_alt, x_alt + self.kutu_genislik, self.y_alt + self.kutu_yukseklik])

    def analiz_et(self):
        print("🚀 Analiz motoru başlatıldı...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            # Araç tespiti
            results = self.model(frame, classes=[2, 3, 5, 7], conf=0.30, verbose=False)
            arac_merkezleri = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    arac_merkezleri.append((cx, cy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Doluluk analizi
            dolu_sayisi = 0
            for (px1, py1, px2, py2) in self.park_alanlari:
                dolu = any(px1 < cx < px2 and py1 < cy < py2 for (cx, cy) in arac_merkezleri)
                
                # Çizim: Kırmızı (Dolu) / Yeşil (Boş)
                renk = (0, 0, 255) if dolu else (0, 255, 0)
                cv2.rectangle(frame, (px1, py1), (px2, py2), renk, 3)
                if dolu: dolu_sayisi += 1

            # Gösterge Paneli
            cv2.rectangle(frame, (0, 0), (500, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"DOLU: {dolu_sayisi} | BOS: {16 - dolu_sayisi}", 
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            self.out.write(frame)

        self.cap.release()
        self.out.release()
        print("✅ İŞLEM TAMAMLANDI! 'FINAL_16_PARK.mp4' dosyasını indirebilirsin.")

# Sistemi çalıştır
if __name__ == "__main__":
    sistem = OtoparkAnalizSistemi(model_path='yolov8n.pt', video_path='parkvideo.mp4', output_path='FINAL_16_PARK.mp4')
    sistem.analiz_et() al kod
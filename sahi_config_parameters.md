# SAHI Uygulamasında Kullanılan Tüm Config Parametreleri

SAHI (Slicing Aided Hyper Inference), büyük görüntülerde küçük nesne tespiti için kullanılan bir framework'tür. Bu dokümanda SAHI'de kullanılan tüm config parametreleri detayıyla listelenmiştir.

## 1. Slicing (Dilimlemme) Parametreleri

### Temel Dilim Parametreleri
- **`slice_height`** (int): Dilim yüksekliği (piksel cinsinden)
  - Örnek: `256`, `512`, `640`
  - Varsayılan: Görüntü boyutuna bağlı

- **`slice_width`** (int): Dilim genişliği (piksel cinsinden)
  - Örnek: `256`, `512`, `640`
  - Varsayılan: Görüntü boyutuna bağlı

### Overlap (Örtüşme) Parametreleri
- **`overlap_height_ratio`** (float): Yükseklik için örtüşme oranı
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.1`, `0.2`, `0.3`
  - Varsayılan: `0.2`

- **`overlap_width_ratio`** (float): Genişlik için örtüşme oranı
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.1`, `0.2`, `0.3`
  - Varsayılan: `0.2`

- **`overlap_ratio`** (float): Hem yükseklik hem genişlik için tek örtüşme oranı
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.2`

## 2. Model Parametreleri

### Model Yolu ve Konfigürasyon
- **`model_path`** (str): Model dosyasının yolu
  - Örnek: `"path/to/model.pt"`, `"yolo11n.pt"`

- **`model_config_path`** (str): Model konfigürasyon dosyasının yolu
  - MMDetection için config dosyası yolu

- **`model_device`** (str): Çıkarım için kullanılacak cihaz
  - Seçenekler: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`
  - Varsayılan: `"cpu"`

### Model Güven Parametreleri
- **`model_confidence_threshold`** (float): Model güven eşiği
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.25`, `0.5`, `0.7`
  - Varsayılan: `0.25`

- **`iou_threshold`** (float): IoU (Intersection over Union) eşiği
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.25`, `0.5`, `0.7`
  - Varsayılan: `0.5`

## 3. Postprocess (İşlem Sonrası) Parametreleri

### Postprocess Türü
- **`postprocess_type`** (str): İşlem sonrası birleştirme türü
  - Seçenekler: `"UNIONMERGE"`, `"NMS"`
  - Varsayılan: `"UNIONMERGE"`

### Postprocess Eşleştirme
- **`postprocess_match_metric`** (str): Eşleştirme metriği
  - Seçenekler: `"IOU"` (Intersection over Union), `"IOS"` (Intersection over Smaller)
  - Varsayılan: `"IOU"`

- **IOU** : Returns the ratio of intersection area to the union

- **IOS** : Returns the ratio of intersection area to the smaller box's area

- **`postprocess_match_threshold`** (float): Eşleştirme eşiği
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.5`, `0.6`, `0.7`
  - Varsayılan: `0.5`

- **`postprocess_class_agnostic`** (bool): Sınıf bağımsız eşleştirme
  - Seçenekler: `True`, `False`
  - Varsayılan: `False`

## 4. Çıktı ve Kaydetme Parametreleri

### Çıktı Dizinleri
- **`project`** (str): Proje dizini
  - Örnek: `"runs/predict"`

- **`name`** (str): Çalıştırma adı
  - Örnek: `"exp1"`, `"test_run"`

- **`source`** (str): Kaynak görüntü/dizin yolu
  - Örnek: `"image.jpg"`, `"images/"`, `"video.mp4"`

### Kaydetme Seçenekleri
- **`save_dir`** (str): Sonuçların kaydedileceği dizin

- **`export_pickle`** (bool): Pickle formatında kaydetme
  - Varsayılan: `False`

- **`export_crop`** (bool): Kırpılmış tahminleri kaydetme
  - Varsayılan: `False`

- **`visual_bbox_thickness`** (int): Görselleştirmede bbox kalınlığı
  - Örnek: `1`, `2`, `3`

- **`visual_text_size`** (float): Görselleştirmede metin boyutu
  - Örnek: `0.3`, `0.5`, `0.8`

- **`visual_text_thickness`** (int): Görselleştirmede metin kalınlığı
  - Örnek: `1`, `2`


## 5. COCO Dataset Parametreleri

### COCO Slicing
- **`ignore_negative_samples`** (bool): Negatif örnekleri yoksay
  - Varsayılan: `False`

- **`out_dir`** (str): Çıktı dizini

- **`train_split_rate`** (float): Eğitim veri bölümleme oranı
  - Değer aralığı: 0.0 - 1.0
  - Örnek: `0.8`, `0.9`

### COCO Evaluation
- **`dataset_json_path`** (str): COCO dataset JSON dosya yolu

- **`result_json_path`** (str): Sonuç JSON dosya yolu

- **`type`** (str): Değerlendirme türü
  - Seçenekler: `"bbox"`, `"segm"`

## 6. Video İnference Parametreleri

- **`video_path`** (str): Video dosya yolu

- **`frame_skip_interval`** (int): Atlanan frame sayısı
  - Örnek: `0`, `1`, `5`

## 7. FiftyOne Entegrasyon Parametreleri

- **`dataset_name`** (str): FiftyOne dataset adı

- **`launch_fiftyone`** (bool): FiftyOne UI'ı başlat
  - Varsayılan: `True`

## 8. Gelişmiş Parametreler

### Performans Optimizasyonu
- **`auto_slice_resolution`** (bool): Otomatik dilim çözünürlüğü
  - Varsayılan: `True`

- **`slice_inference_batch_size`** (int): Dilim çıkarım batch boyutu
  - Örnek: `1`, `4`, `8`

### Hata Ayıklama
- **`verbose`** (int): Detaylı çıktı seviyesi
  - Seçenekler: `0`, `1`, `2`

- **`return_dict`** (bool): Dictionary olarak sonuç döndür
  - Varsayılan: `True`

## Örnek Kullanım

### CLI Komutu
```bash
sahi predict \
    --model_path "yolo11n.pt" \
    --model_confidence_threshold 0.25 \
    --source "test_image.jpg" \
    --slice_height 640 \
    --slice_width 640 \
    --overlap_height_ratio 0.2 \
    --overlap_width_ratio 0.2 \
    --postprocess_type "UNIONMERGE" \
    --postprocess_match_threshold 0.5 \
    --export_pickle \
    --export_crop
```

### Python API
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Model yükleme
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolo11n.pt",
    confidence_threshold=0.25,
    device="cpu"
)

# Sliced prediction
result = get_sliced_prediction(
    image="test_image.jpg",
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="UNIONMERGE",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False,
    export_pickle=True,
    export_crop=True
)
```

## Notlar

- Parametreler framework'e (YOLOv8, MMDetection, vb.) göre değişiklik gösterebilir
- Optimal parametre değerleri görüntü boyutuna ve nesne türüne bağlıdır
- Slice boyutları GPU belleğine göre ayarlanmalıdır
- Overlap oranları tespit kalitesini artırır ancak işlem süresini uzatır
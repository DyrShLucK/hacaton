import datetime

# Аутентификация в Google Earth Engine
import ee
import ee.mapclient
import geemap
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

project_id = '' # id для EE

# Аутентификация и инициализация Earth Engine
ee.Authenticate()
ee.Initialize(project=project_id)

# --- ПАРАМЕТРЫ ДЛЯ ЗАМЕНЫ ---
GCS_BUCKET = 'ваш-бакет'  # Например: 'my-gee-project-tiffs'
TIFF_FOLDER = 'tiffs'  # Папка в бакете с TIFF
METADATA_CSV = 'metadata.csv'  # CSV с полями: filename, date, cloudy_pixel_percentage

# Аутентификация
ee.Authenticate()
ee.Initialize(project='crucial-bloom-239020')

# === 1. Создание ImageCollection из локальных TIFF ===
# Загрузка метаданных из CSV в Google Cloud Storage
metadata = ee.FeatureCollection(f'gs://{GCS_BUCKET}/{METADATA_CSV}')


def create_image(feature):
    """Преобразует запись из CSV в Earth Engine Image"""
    filename = feature.get('filename')
    cloud_percent = feature.getNumber('cloudy_pixel_percentage')
    date_str = feature.get('date')

    return ee.Image.loadGeoTIFF(f'gs://{GCS_BUCKET}/{TIFF_FOLDER}/{filename}') \
        .set('system:time_start', ee.Date(date_str).millis()) \
        .set('CLOUDY_PIXEL_PERCENTAGE', cloud_percent)


# Создаем коллекцию изображений
custom_collection = ee.ImageCollection(metadata.map(create_image))

# === 2. Остальная часть кода с адаптацией ===
aoi = ee.Geometry.Point([63.0429801940917, 61.9103202819824])

# --- ВАЖНО: Укажите реальные названия каналов в ваших TIFF ---
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Замените на реальные имена каналов!

# Фильтрация и подготовка данных
filtered_collection = custom_collection \
    .filterDate('2023-05-01', '2023-07-31') \
    .filterBounds(aoi) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .map(lambda img: img.select(BANDS))  # Выбор нужных каналов

composite = filtered_collection.median()

# --- Обучение модели (оставшаяся часть вашего кода) ---
# Загрузка полигонов свалок
gdf = gpd.read_file('svalki.gpkg', layer='svalki')
features = geemap.geopandas_to_ee(gdf)

# Подготовка обучающих данных
training = composite.sampleRegions(
    collection=features,
    properties=['param'],  # Убедитесь, что это правильное имя столбца
    scale=10
)

# Разделение данных и обучение модели
classifier = ee.Classifier.smileRandomForest(10).train(
    features=training.randomColumn().filter(ee.Filter.lt('random', 0.7)),
    classProperty='param',
    inputProperties=BANDS
)

# Классификация и экспорт
classified = composite.classify(classifier)
geemap.ee_export_vector(
    classified.eq(0).reduceToVectors(geometry=aoi, scale=10),
    'landfills.geojson'
)
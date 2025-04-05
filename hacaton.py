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

project_id = '' #id для EE

# Аутентификация и инициализация Earth Engine
ee.Authenticate()
ee.Initialize(project=project_id)

# 1. Определение области интереса (AOI) и даты
aoi = ee.Geometry.Point([63.0429801940917, 61.9103202819824])

start_date = '2023-05-01'
end_date = '2023-07-31'

# 2. Загрузка и подготовка обучающих данных (полигоны свалок из GeoPackage)

# Путь к GeoPackage (загрузите его в Colab или Google Drive)
gpkg_path = 'svalki.gpkg'  # Замените на фактический путь
layer_name = 'svalki'       # Замените на имя слоя с полигонами свалок в GeoPackage

gdf = gpd.read_file(gpkg_path, layer=layer_name)
print(gdf.head()) # Проверьте, что данные загрузились правильно

# Убедитесь, что у вас есть столбец с метками классов (например, 'class', 'landfill', 'is_landfill')
# Если нет, создайте его на основе других атрибутов, если необходимо
# Пример:  Предположим, что у вас есть столбец 'type', где 'landfill' означает свалку
# gdf['class'] = gdf['type'].apply(lambda x: 1 if x == 'landfill' else 0)

# Проверка наличия столбца 'class'
if 'param' not in gdf.columns:
    print("Error: Column 'class' not found in GeoPackage.  Please add a 'class' column (1 for landfill, 0 for not landfill).")
    exit()

# b) Преобразование GeoDataFrame в FeatureCollection Earth Engine

def geojson_to_ee_feature(feature):
    """Converts a GeoJSON Feature to an Earth Engine Feature and casts 'class' to int."""
    geom = feature['geometry']
    props = feature['properties']
    # Преобразуем 'class' в целое число
    try: # если class это int или float
        props['param'] = int(props['param'])
    except ValueError: # если class это строка
        try:
            props['param'] = int(float(props['param']))
        except ValueError:
            print(f"can't convert param label {props['param']} to integer")
            raise # кидаем оригинальную ошибку

    return ee.Feature(ee.Geometry(geom), props)

# Преобразование GeoDataFrame в GeoJSON
geojson = gdf.__geo_interface__

# Создание списка ee.Feature из GeoJSON
ee_features = [geojson_to_ee_feature(f) for f in geojson['features']]

# Создание FeatureCollection из списка ee.Feature
features = ee.FeatureCollection(ee_features)

# ВАЖНО: Проверка FeatureCollection - убедитесь, что столбец 'class' существует
first_feature = features.first()
print("First feature properties:", first_feature.getInfo()['properties'])

# 3. Выбор снимков ДЗЗ (Sentinel-2 или Landsat)
# Пример использования Sentinel-2:
image_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterDate(start_date, end_date)
                    .filterBounds(aoi)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))  # Фильтр по облачности

# Выбор нужных каналов (bands) - определите после фильтрации!
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'] # Sentinel-2 bands: Blue, Green, Red, NIR, SWIR1, SWIR2

# Обработка изображений с разными наборами каналов (при необходимости)
def select_bands(image):
    return image.select(bands)

image_collection = image_collection.map(select_bands)

# Выбор медианного изображения (или другого подходящего способа композиции)
composite = image_collection.median()  # Теперь composite должен иметь только нужные bands
print(f"composite bands: {composite.bandNames().getInfo()}")  # Печать, чтобы убедиться в наборе каналов

# 4. Подготовка признаков (features) для машинного обучения
# ВАЖНО: Укажите правильное имя столбца с классами!
correct_class_name = 'param'  # <----- ЗАМЕНИТЕ НА ПРАВИЛЬНОЕ ИМЯ СТОЛБЦА

training = composite.sampleRegions(
    collection=features,
    properties=[correct_class_name],  # <----- ИСПОЛЬЗУЙТЕ ПРАВИЛЬНОЕ ИМЯ СТОЛБЦА
    scale=10  # Разрешение пикселя (Sentinel-2: 10m, Landsat: 30m)
)

# 5. Разделение данных на обучающую и тестовую выборки
training_data = training.randomColumn()
train_set = training_data.filter(ee.Filter.lt('random', 0.7))
test_set = training_data.filter(ee.Filter.gte('random', 0.7))

# **ОТЛАДКА: ПРОВЕРКА РАСПРЕДЕЛЕНИЯ КЛАССОВ**
print("Train set class distribution:", train_set.aggregate_histogram(correct_class_name).getInfo()) #  <----- ИСПОЛЬЗУЙТЕ ПРАВИЛЬНОЕ ИМЯ СТОЛБЦА
print("Test set class distribution:", test_set.aggregate_histogram(correct_class_name).getInfo()) #  <----- ИСПОЛЬЗУЙТЕ ПРАВИЛЬНОЕ ИМЯ СТОЛБЦА

# 6. Обучение модели машинного обучения (Random Forest)
classifier = ee.Classifier.smileRandomForest(10)  # 10 деревьев
classifier = classifier.train(
    features=train_set,
    classProperty=correct_class_name,  #  <----- ИСПОЛЬЗУЙТЕ ПРАВИЛЬНОЕ ИМЯ СТОЛБЦА
    inputProperties=bands
)

# 7. Оценка точности модели
test_accuracy = test_set.classify(classifier).errorMatrix(correct_class_name, correct_class_name) #  <----- ИСПОЛЬЗУЙТЕ ПРАВИЛЬНОЕ ИМЯ СТОЛБЦА
print('Матрица ошибок теста: ', test_accuracy.getInfo())
print('Точность теста: ', test_accuracy.accuracy().getInfo())

# 8. Классификация изображения
classified_image = composite.classify(classifier)
classified_image = classified_image.toInt()  # <--- Добавлено: Явное преобразование в int


# 9. Векторизация результатов (пикселей со значением 0)
landfill_mask = classified_image.eq(0)  # Создаем бинарную маску для свалок

# Преобразование растра в векторные полигоны
aoi_polygon = aoi.buffer(10000).bounds()  # Создаем область вокруг точки (1 км буфер)
vectors = landfill_mask.reduceToVectors(
    geometry=aoi_polygon,
    scale=10,
    maxPixels=1e13,
    geometryType='polygon',
    eightConnected=False  # Используем 4-связность для более компактных полигонов
)

# 10. Экспорт в GeoJSON
output_file = 'landfills.geojson'
geemap.ee_export_vector(vectors, output_file, verbose=True)

print(f"Векторные полигоны сохранены в: {output_file}")
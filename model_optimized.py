import os
import pickle
from numpy.linalg import norm
from tensorflow.keras.preprocessing.image import load_img
import pandas as pd
from annoy import AnnoyIndex
import time
import numpy as np
from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(ROOT_DIR, 'static\\fashion_clear')
extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def get_file_list(root_dir):
    file_list = []

    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))

    return file_list


images_dir_path = sorted(get_file_list(IMAGES_PATH))
print(len(images_dir_path))

resnet_model = 'pickles\\features_resnet.pkl'


# vgg_model = 'pickles\\features_vgg16.pkl'


def load_model_pickle(_model_path):
    print("Model use : ", _model_path)
    my_list = pickle.load(open(os.path.join(ROOT_DIR, _model_path), 'rb'))
    return my_list, _model_path


feature_list, model_path = load_model_pickle(resnet_model)

file_names = []
for i in range(len(images_dir_path)):
    name = images_dir_path[i].split('\\')[-1]
    file_names.append(name)

df = pd.DataFrame({'img_name': file_names, 'features': feature_list})

TREE_SIZE = 75
RECOMMENDED_ITEM_SIZE = 5


def annoy_search(ref_index, features, tree_size, recommended_item_size, _metric):
    start = time.time()
    index = df.loc[ref_index]['features']
    f = len(index)
    t = AnnoyIndex(f, metric=_metric)
    print("Metric use: ", _metric)

    for e in range(len(features)):
        t.add_item(e, features[e])

    t.build(tree_size, n_jobs=-1)
    similar_img_ids, distances = t.get_nns_by_item(ref_index, recommended_item_size, include_distances=True)
    end = time.time()
    annoy_runtime = end - start
    print("ANNOY Runtime: ", end - start)
    return similar_img_ids, distances, annoy_runtime


def extract_feature_from_uploaded_image(image_path):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    img = load_img(image_path)
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    expanded_img_array = np.array(expanded_img_array, dtype='float64')

    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)

    image_path = image_path.split('\\')[-1]

    df.loc[len(df.index)] = [image_path, normalized_features]
    feature_list.append(normalized_features)
    REF_INDEX = len(df.index) - 1

    similar_img_ids, _, __ = annoy_search(ref_index=REF_INDEX, features=feature_list, tree_size=TREE_SIZE,
                                          recommended_item_size=RECOMMENDED_ITEM_SIZE, _metric='euclidean')

    clear_recommendation_data()
    return df.iloc[similar_img_ids[1:]]['img_name'].tolist()


def clear_recommendation_data():
    global df, feature_list
    df = df.iloc[:-1, :]
    feature_list = feature_list[:-1]

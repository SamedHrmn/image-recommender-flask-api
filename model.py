from tensorflow.keras.applications import vgg16
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random


def get_images(uploaded_image_path , recommendation_threshold):
    categories = {
        '0': 'Ayakkabı',
        '1': 'Ruj',
        '2': 'El Çantası',
        '3': 'Parlatıcı',
        '4': 'Kolye',
        '5': 'Saat',
        '6': 'Yüzük',
        '7': 'Bilezik',
        '8': 'Bot',
        '9': 'Küpe'
    }

    def get_key_category(val):
        for v in categories:
            if val == categories[v]:
                return str(v)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    imgs_path = os.path.join(ROOT_DIR,"static")

    model = vgg16.VGG16(weights='imagenet', pooling='max')

    for layer in model.layers:
        layer.trainable = False

    feat_extractor = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

    files = [imgs_path + "\\" + x for x in os.listdir(imgs_path) if "png" in x]

    shoe_list = []

    for i in files:
        splitted = i.split('\\')[-1]
        if splitted.split('_')[1] == get_key_category('Ayakkabı'):
            shoe_list.append(i)

    resimler = []
    resimler_path = []
    referans_img_path = uploaded_image_path
    referans_img = load_img(referans_img_path, target_size=(224, 224))
    referans_img_path = referans_img_path.split('\\')[-1]
    resimler_path.append(referans_img_path)
    referans_img_batch = np.expand_dims(img_to_array(referans_img), axis=0)
    resimler.append(referans_img_batch)

    similar_random_img_list = {}
    similar_img = []
    index = 0

    def get_similar_random_image(count_similarity, i, step, src_count):
        sim = count_similarity
        print("Sim counter: " + str(sim) + " src_counter: " + str(src_count))
        src_count += 1
        if src_count >= 10:
            print("Lutfen baska resim deneyiniz.")
            return []

        if sim < 1:
            path = shoe_list[random.randint(0 + step, len(shoe_list) - 1)]
            img = load_img(path, target_size=(224, 224))
            path = path.split('\\')[-1]
            batch = np.expand_dims(img_to_array(img), axis=0)
            similar_img.append(batch)

            resV = np.vstack(resimler)
            simV = np.vstack(similar_img)
            processed_resV = preprocess_input(resV.copy())
            processed_simV = preprocess_input(simV.copy())
            features_resV = feat_extractor.predict(processed_resV)
            features_simV = feat_extractor.predict(processed_simV)

            similarity = cosine_similarity(features_resV, features_simV[[index + i]])
            print("Sim score : ", str(similarity))

            if similarity >= recommendation_threshold:
                print("Girdi")
                dic = {path: similarity}
                similar_random_img_list.update(dic)
                sim = sim + 1
            get_similar_random_image(count_similarity=sim, i=i + 1, step=1, src_count=src_count)
        else:
            return similar_random_img_list

    get_similar_random_image(count_similarity=0, i=0, step=2, src_count=0)
    print(similar_random_img_list)

    cos_similarities_df = pd.read_pickle(os.path.join(ROOT_DIR,"dataframe.pkl"))
    path_prefix = 'D:\\Projelerim\\ML\\deneme\\images\\'

    if len(similar_random_img_list) > 0:
        keys = []
        values = []
        for key in similar_random_img_list:
            keys.append(key)
            values.append(similar_random_img_list[key])

        a = pd.DataFrame(cos_similarities_df[path_prefix + str(keys[0])]).sort_values(by=path_prefix + str(keys[0]),
                                                                                      ascending=False)
        a.rename(columns=lambda s: s.split('\\')[-1], index=lambda s: s.split('\\')[-1], inplace=True)

        rec_list = []
        for i in range(0, 5):
            eleman = list(a.iloc[[i]].index)
            rec_list.append(eleman)

    return rec_list

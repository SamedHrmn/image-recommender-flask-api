# image-recommender-flask-api

This project written as a backend service. 

## Content
This service gives recommendation with transfer learning based by using uploaded fashion images. For able to use the model, you must have the features of the images.<br>
Service uses extracted features by ResNet50 or VGG16 architecture. You can extract own features via ResNet50 like this example below. <br>

```python
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
from numpy.linalg import norm

resnet_model = ResNet50(weights='imagenet',include_top=False)
print(resnet_model.summary())


def feature_extraction(img_path , model):
  img = load_img(img_path)
  img_array = img_to_array(img)

  expanded_img_array = np.expand_dims(img_array, axis=0)
  expanded_img_array = np.array(expanded_img_array,dtype='float64')

  preprocessed_img = preprocess_input(expanded_img_array)
  features = model.predict(preprocessed_img)
  flattened_features = features.flatten()
  normalized_features = flattened_features / norm(flattened_features)
  return normalized_features
```

<br><br>

![Ekran görüntüsü 2021-07-01 154223](https://user-images.githubusercontent.com/60006881/124126103-fa9b0880-da82-11eb-9616-2ca22742712d.png)


## Licence

MIT License

Copyright (c) 2021 SamedHrmn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

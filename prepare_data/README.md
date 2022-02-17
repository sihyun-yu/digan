## Kinectics Food

### Overview

[Kinetics-600](https://arxiv.org/abs/1808.01340) is a large-scale 600-class video action dataset, consists of a total of 495,547 videos. We sub-sampled a food subclass in the dataset to train the model, where we follow the list of such a subclass from [Weissenborn et al. (2020)](https://openreview.net/forum?id=rJgsskrFwH), namely: `baking, barbequing, breading, cooking, cutting, pancake, vegetables, meat, cake, sandwich, pizza, sushi, tea, peeling, fruit, eggs, salad`. We use train split for the model training and use the validation set for the evaluation. Note that we only use these classes, different from [Weissenborn et al. (2020)](https://openreview.net/forum?id=rJgsskrFwH) to train the model with the whole dataset.

### Download and extract videos
```
# download and extract train file
bash download_extract.sh train

# download and extract validation file
bash download_extract.sh val
```

### Preprocess into png
```
python preprocess.py
```
After running all codes, one can obtain png preprocessed Kinectics-food dataset at `./train` and `./val` folders.

### Reference
We used the downloading and extraction codes from the following repository: [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset).

# Codebase for Master's Thesis: Image-Based Query Search Engine via Deep Learning

**Link to the thesis**: https://repository.tudelft.nl/islandora/object/uuid:4a2c9c6f-b2b8-41d6-9b70-69c4f246c964?collection=education

**Note**: This thesis contributes to a module (nearest neighbour search) in an image-query based search engine designed and implemented by 3 MSc students. This codebase only contains the codes that are relevant to the thesis. The codebase of the final search engine: https://github.com/YYao-42/Image-Search-Engine-for-Historical-Research. If you want to know more about the whole engine and not just the nearest neighbour search part, I recommend you check out that repository instead, which has more detailed information.

### Abstract
Typically, people search images by text: users enter keywords and a search engine returns relevant results. However, this pattern has limitations. An obvious drawback is that when searching in one language, users may miss results labelled in other languages. Moreover, sometimes people know little about the object in the image and thus would not know what keywords could be used to search for more information. Driven by this use case with many applications, content-based image retrieval (CBIR) has recently been put under the spotlight, which aims to retrieve similar images in the database solely by the content of the query image without relying on textual information.

To achieve this objective, an essential part is that the search engine should be able to interpret images at a higher level instead of treating them simply as arrays of pixel values. In practice, this is done by extracting distinguishable features. Many effective algorithms have been proposed, from traditional handcrafted features to more recent deep learning methods. Good features may lead to good retrieval performance, but the problem is still not fully solved. To make the engine useful in real-world applications, retrieval efficiency is also an important factor to consider while has not received as much attention as feature extraction.

In this work, we focus on retrieval efficiency and provide a solution for real-time CBIR in million-scale databases. The feature vectors of database images are extracted and stored offline. During the online procedure, such feature vectors of query images are also extracted and then compared with database vectors, finding the nearest neighbours and returning the corresponding images as results. Since feature extraction only performs once for each query, the main limiting factor of retrieval efficiency in large-scale database is the time of finding nearest neighbours. Exact search has been shown to be far from adequate, and thus approximate nearest neighbour (ANN) search methods have been proposed, which mainly fall into two categories: compression-based and tree/graph-based. However, these two types of approaches are usually not discussed and compared together. Also, the possibility of combining them has not been fully studied. Our study (1) applies and compares methods in both categories, (2) reveals the gap between toy examples and real applications, and (3) explores how to get the best of both worlds. Moreover, a prototype of our image search engine with GUI is available on https://github.com/YYao-42/ImgSearch.

![cover](https://user-images.githubusercontent.com/39213403/181117170-68f8ef03-4276-40f8-984c-d2a0077b3a1b.png)

### How to run
- Run `main_train.py` to download datasets and pretrained networks. You can also retain the network by specifying relevant parameters.
- Run `main_retrieve.py` to generate feature vectors. E.g.,
  ```
  python3 -m src.main_retrieve --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --datasets 'oxford5k,paris6k,roxford5k,rparis6k'
  ```
  You can extract features of datasets other than (r)oxford and (r)paris by using function `extr_selfmade_dataset`. See line 404-409.
- Run `test_name of the dataset.py` to get the mAP and the matching time of all methods.
  - To report mAP, use K = n_database; To compare the matching time, use K = 100.
  - Set `ifgenerate=True` on the first run whenever K changes. Then set `ifgenerate=False`.
- Run `server.py`to show the GUI. E.g.,
  ```
  python3 -m src.server --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --datasets 'oxford5k,paris6k,Andrea,flickr100k,Custom,GLM'
  ```
 

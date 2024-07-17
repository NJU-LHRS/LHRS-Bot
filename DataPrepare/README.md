# Notice
We recommend users download the image using the provided coordinates from the Google Earth API. Typically, each image is organized with the following naming format:

```
Country_Id_City_ULLon_ULLat_LRLon_LRLat.jpg

# Country: Name of the country
# Id: Unique identifier for this city
# City: Name of the city
# ULLon: Longitude of the upper left corner
# ULLat: Latitude of the upper left corner
# LRLon: Longitude of the lower right corner
# LRLat: Latitude of the lower right corner

```

⚠️ We recommand user to download from the data following data source:
+ RSVQA: [source](https://rsvqa.sylvainlobry.com/)
+ NWPU: [source](https://huggingface.co/datasets/timm/resisc45)
+ RSICD: [source](https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file)
+ RSITMD: [source](https://drive.google.com/file/d/1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6/view)
+ DIOR-RSVG: [source](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_)
+ RSVG: [source](https://drive.google.com/file/d/1kgnmVC6FVKdxCwaoG77sOfkaIHS_XiFt/view)
+ UCM: [source](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
+ fMoW: [source](https://github.com/fMoW/dataset)
+ LLaVA: Follow instruction from [here](https://github.com/haotian-liu/LLaVA/blob/main/README.md#visual-instruction-tuning) (we only use random 20K subset).


# Pretraining
+ 	Please download all the data through the Google Earth API and place it into a single directory with a name ending in “_Image.”

+ Download our caption files from [here](https://huggingface.co/datasets/PumpkinCat/LHRS_Data/tree/main/Stage1) and place all the json files in a folder named “OSMCapAnn.”

Finally, your pretraining data folder should be structured as follows:
```
|-PretrainData
|----XXX_Image
|    |---xxxxx.jpg
|    ...
|----OSMCapAnn
|    |features_01.json
|    ...
```

# SFT
## Stage2
+ Download the instruction data from [here](https://huggingface.co/datasets/PumpkinCat/LHRS_Data/tree/main/Stage2) and the corresponding images. Then organize the image folder names and json names in the following similar format:

```
|-Stage2Data
|----RSITMD_Image
|----RSITMD.json
|----RSITMDDetail_Image
|----RSITMDDetail.json
|----UCM_Image
|----UCM.json
|    ...
```

## Stage3
+ Download the instruction data from [here](https://huggingface.co/datasets/PumpkinCat/LHRS_Data/tree/main/Stage3) and the corresponding images. Then organize the image folder names and json names in the following similar format:

```
|-Stage3Data
|----OSM_Image
|----OSM.json
|----LLaVA_Image
|----LLaVA.json
|    ...
```

# Evaluation
+ Classification, VQA are using the same format. Therefore, just download from source.
+ Our reformat result of VG can be found at [here](https://huggingface.co/datasets/PumpkinCat/LHRS_Data).



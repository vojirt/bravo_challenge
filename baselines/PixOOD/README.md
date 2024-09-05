# PixOOD method
This is implementation of the "PixOOD: Pixel-Level Out-of-Distribution
Detection" ECCV 2024 method into the BRAVO challenge toolkit. The code is
almost 1-to-1 copy from the official
[github](https://github.com/vojirt/PixOOD), except for:
1. quality of life changes so it is easily run within this benchmark
2. new decoder architectures that was submitted specifically to the BRAVO Challenge 2024

You can find the details about the method in:
```latex
@InProceedings{Vojir_2024_ECCV,
    author    = {Vojíř, Tomáš and Šochman, Jan and Matas, Jiří},
    title     = {{PixOOD: Pixel-Level Out-of-Distribution Detection}},
    booktitle = {ECCV},
    year      = {2024},
}
```

and about the submissions and its variants in (**TBD**):
*for now you can see the submitted technical paper [here](https://drive.google.com/file/d/1_Wa-ywXNwgDNJgXEU8wJm3MXDxdAR3wo/view)*
```latex
@InProceedings{UNCV_workshop_paper_2024,
    author    = {},
    title     = {{}},
    booktitle = {ECCV},
    year      = {2024},
}
```

## Running the method
There are four variants of the method. First download the pre-trained checkpoints:
```
cd chkpts
chmod +x download_chkpts.sh
./download_chkpts.sh
```

Install required packages (or optionally first create python virtual environment of your choosing):
```
python -m pip install -r requirements.txt
```

To run individual variants, pass the variant name as the parameter to the evaluation script, e.g.:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_ood_bravo.py --method PixOOD
```
The method can be set to: `PixOOD, PixOOD_Dec, PixOOD_Dec_CityBDD, PixOOD_R101DLv3`.
You can also pass `--verbose` to the command above to save estimated segmentation masks and anomaly scores.

## Licence
Copyright (c) 2024 Toyota Motor Europe<br>
Patent Pending. All rights reserved.

This work is licensed under a [Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc/4.0/)

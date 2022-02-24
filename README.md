# RelationaFutureCaptioningModel (RFCM)

<p align="center"><img src="assets/logo.png" alt="Logo" title="Logo" /></p>

This code implements the Relational Future Captioning Model (RFCM).

## Installation

We use `python=3.8.5` and `pytorch=1.7.1`. Tested on `Ubuntu 18.04`.

~~~bash
git clone https://github.com/keio-smilab22/RelationalFutureCaptioningModel.git
cd RelationalFutureCaptioningModel
pip install -r requirements.txt
~~~
If you have problems with the `pycocoevalcap` package try uninstalling it and installing it with this command instead: `pip install git+https://github.com/salaniz/pycocoevalcap`
The METEOR metric requires `java`. Either install the latest Java 1.8 through the system (Tested with `Java RE 1.8.0_261`) or install with conda `conda install openjdk`. Make sure your locale is set correct i.e. `echo $LANG` outputs `en_US.UTF-8`
Download and extract for YooCook2-FC dataset: [COOT output Embeddings](https://drive.google.com/file/d/1atbI9HaFArNPeZzkvrJ9TnkCAal6gyUQ/view?usp=sharing) ~230mb, [Pretrained Captioning models](https://drive.google.com/file/d/1IV85_DXWx1SJL9ZJuT6Qvvyx8obE9f9x/view?usp=sharing) ~540 mb

~~~bash
tar -xzvf provided_embeddings.tar.gz
tar -xzvf provided_models_caption.tar.gz
~~~

## Precompute all the text features

~~~bash
python data_read_youcook2_meta.py
python precompute_text.py youcook2 --cuda
~~~


### Train and validate RFCM on COOT embeddings

~~~bash
# YouCook2-FC
# Train RFCM on COOT clip embeddings
python train_caption.py -c config/caption/paper2020/yc2_100m_coot_clip_mart.yaml

# show trained results
python show_caption.py -m base

# evaluate provided models
python train_caption.py -c config/caption/paper2020/yc2_100m_coot_vidclip_mart.yaml --validate --load_model provided_models_caption/yc2_100m_coot_vidclip_mart.pth
# etc.
~~~

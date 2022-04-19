ARRAY=(1 2 3 4)

for num in ${ARRAY[@]}; do
    python train_caption.py -c config/caption/paper2020/bddx_100m_coot_clip_mart.yaml
    echo $num"回目のループです"
done

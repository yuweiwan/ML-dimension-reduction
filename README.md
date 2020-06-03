# ML-dimension-reduction
PCA and LLE  

command:  
python3 main.py --knn-k 5 --test-split data/dev --predictions-file simple-knn-preds.txt
python3 main.py --knn-k 5 --dr-algorithm lle --lle-k 10 --target-dim 300 --test-split dev --predictions-file pca-knn-preds.txt  
python3 accuracy.py --labeled data/labels-mnist-dev.npy --predicted simple-knn-preds.txt

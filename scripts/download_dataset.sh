export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main
export IMAGES_TARGET=../data/images
mkdir -p $IMAGES_TARGET

wget $SRC/lm/lm_base.zip -P $IMAGES_TARGET # Base archive with dataset info, camera parameters, etc.
wget $SRC/lm/lm_models.zip -P $IMAGES_TARGET # 3D object models.
wget $SRC/lm/lm_test_all.zip -P $IMAGES_TARGET # All test images ("_bop19" for a subset used in the BOP Challenge 2019/2020).
wget $SRC/lm/lm_train_pbr.zip -P $IMAGES_TARGET # PBR training images (rendered with BlenderProc4BOP).


cd $IMAGES_TARGET
unzip lm_base.zip             # Contains folder "lm".
unzip lm_models.zip -d $IMAGES_TARGET     # Unpacks to "lm".
unzip lm_test_all.zip -d $IMAGES_TARGET   # Unpacks to "lm".
unzip lm_train_pbr.zip -d $IMAGES_TARGET  # Unpacks to "lm".
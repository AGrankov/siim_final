export CUDA_VISIBLE_DEVICES=0
export DATA_INPUT="../../siim_final/input/stage_2_images"
export SAMPLE_SUBMISSION="../../siim_final/input/stage_2_sample_submission.csv"
export OUTPUT_FILENAME="../submissions/final_1_sub.csv.gz"

python3 binary/predicting/predict_efn_orig.py -it $DATA_INPUT -ss $SAMPLE_SUBMISSION -o ../p_input/binary/efn0_1024_d75_he_v1.csv -kf 6 -mp ../models/binary/efficientnet0_is1024_dropout75_he -ms 0 -is 1024
python3 binary/predicting/predict_efn_orig.py -it $DATA_INPUT -ss $SAMPLE_SUBMISSION -o ../p_input/binary/efn5_512_d90_he_v1.csv -kf 4 -mp ../models/binary/efficientnet5_is512_dropout90_he -ms 5 -is 512 -bs 2
python3 binary/predicting/predict_efn_add.py -it $DATA_INPUT -ss $SAMPLE_SUBMISSION
python3 binary/predicting/predict_efn_dynamic_aug.py -it $DATA_INPUT -ss $SAMPLE_SUBMISSION
python3 binary/predicting/predict_efn_cropped.py -it $DATA_INPUT -ss $SAMPLE_SUBMISSION

python3 predicting/predict_segmentation.py -i $DATA_INPUT -ss $SAMPLE_SUBMISSION -st 0.65 -mft 1500 -o $OUTPUT_FILENAME -uk True

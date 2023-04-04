
EXTRACTOR_TYPE="mfccOnly"
ADDRESS="./vox/"

for MODEL_NAME in  MODEL_NAME ecapatdnn vgg
do
    for RATIO in 0.005 0.01 0.05 0.1 0.2 0.4 0.5 0.6 0.8 1
    do
        poisoned_address="wav/${EXTRACTOR_TYPE}"
        mkdir -p "${poisoned_address}"
        python genPosionXvector.py -d "${ADDRESS}" --extracter_type "${EXTRACTOR_TYPE}"
        if [ $? -ne 0 ]
        then
            echo python genPosionXvector.py -d "${ADDRESS}" --extracter_type "${EXTRACTOR_TYPE}" "COMMAND FAILED"
            exit 1
        fi

        mkdir -p "data/${EXTRACTOR_TYPE}"
        after_poison_address="data/${EXTRACTOR_TYPE}/poison${RATIO}.txt"
        python data/genid.py -d "${ADDRESS}" --extracter_type "${EXTRACTOR_TYPE}" --ratio "${RATIO}"
        if [ $? -ne 0 ]
        then
            echo python data/genid.py -d "${ADDRESS}" --extracter_type "${EXTRACTOR_TYPE}" --ratio "${RATIO}" "COMMAND FAILED"
            exit 1
        fi

        weights_path="./saveCheckPoint/${EXTRACTOR_TYPE}/poison${RATIO}/weight.pth"
        acc_path="./saveCheckPoint/${EXTRACTOR_TYPE}/poison${RATIO}/acc.txt"
        mkdir -p "./saveCheckPoint/${EXTRACTOR_TYPE}/poison${RATIO}"
        python "${MODEL_NAME}/train.py" --after_poison_address "${after_poison_address}"  --weights_path "${weights_path}" --acc_path "${acc_path}"
        if [ $? -ne 0 ]
        then
            echo python "${MODEL_NAME}/train.py" --after_poison_address "${after_poison_address}"  --weights_path "${weights_path}" --acc_path "${acc_path}" "COMMAND FAILED"
            exit 1
        fi
    done
done

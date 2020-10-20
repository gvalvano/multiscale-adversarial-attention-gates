# In this example, we show how to run our model on the ACDC dataset, available under the folder ${dpath}. Results will
# be saved under the folder ${res_dir}. You can monitor the training process running in your bash the command:
# tensorboard --logdir=results/${dset_name}/graphs , and then navigating to the localhost in your browser.
# In this example, we will train the model using 50% of the annotated data. Refer to `data_interface/utils_acdc/split_data`
# for more details on the training volumes. Finally, we test on the ADCD testing volumes.

dpath='../DATA/ACDC'
res_dir='.'
dset_name='acdc'

# Flag for CUDA_VISIBLE_DEVICE:
CUDA_VD=0

for run_id_and_path in \
    'WEAK_ACDC_MultiscaleAAGs_224x224 model'

    do

    # shellcheck disable=SC2086
    set -- ${run_id_and_path}
    run_id=$1
    path=$2

    for perc in 'perc50'
        do for split in 'split0'

            do echo "${run_id}"_${perc}_${split}

            python -m train  --RUN_ID="${run_id}"_${perc}_${split} \
                             --n_epochs=450 \
                             --CUDA_VISIBLE_DEVICE=${CUDA_VD} \
                             --data_path=${dpath} \
                             --experiment="${path}" \
                             --dataset_name=${dset_name} \
                             --verbose=True \
                             --results_dir=${res_dir}\
                             --n_sup_vols=${perc} \
                             --split_number=${split}

            data_path_testing='../DATA/ACDC_testing'
            python -m test_on_acdc_test_set \
                            --RUN_ID="${run_id}"_${perc}_${split} \
                            --CUDA_VISIBLE_DEVICE=${CUDA_VD}  \
                            --data_path=${data_path_testing} \
                            --experiment="${path}" \
                            --dataset_name=${dset_name} \
                            --verbose=False \
                            --results_dir=${res_dir}\
                            --n_sup_vols=${perc} \
                            --split_number=${split}
            done
        done
    done

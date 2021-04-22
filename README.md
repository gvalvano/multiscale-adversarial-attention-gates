
#  Learning to Segment from Scribbles using Multi-scale Adversarial Attention Gates  
  
Code for the paper:  
  
> Valvano, Gabriele, Andrea Leo, and Sotirios A. Tsaftaris. "Learning to Segment from Scribbles using Multi-scale Adversarial Attention Gates." IEEE Transactions on Medical Imaging (2021).
  
The official project page is [here](https://vios-s.github.io/multiscale-adversarial-attention-gates).  
An online version of the paper can be found [here](https://arxiv.org/abs/2007.01152).  

## Citation:  
```  
@ARTICLE{9389796,
  author={Valvano, Gabriele and Leo, Andrea and Tsaftaris, Sotirios A.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Learning to Segment from Scribbles using Multi-scale Adversarial Attention Gates}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2021.3069634}
}
```  
  
<img src="https://github.com/vios-s/multiscale-adversarial-attention-gates/blob/main/images/banner.png" alt="mscale_aags" width="600"/>

----------------------------------  
  
## Notes:  
  
You can find the entire tensorflow model inside `expriments/acdc/model.py`. This file contains the main class that is used to train on the ACDC dataset. Please, refer to the class method `define_model()` to see how to correctly build the CNN architecture. The structure of the segmentor and the discriminator alone can be found under the folder `architectures`.
  
Once you download the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and the [scribble annotations](https://vios-s.github.io/multiscale-adversarial-attention-gates/data), you can pre-process it using the code in the file `data_interface/utils_acdc/prepare_dataset.py`. 
You can also train with custom datasets, but you must adhere to the template required by `data_interface/interfaces/dataset_wrapper.py`, which assumes the access to the dataset is through a tensorflow dataset iterator.

Once preprocessed the data, you can start the training running the command:  
```  
python -m train --RUN_ID="${run_id}"_${perc}_${split} --n_epochs=450 --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${dpath} --experiment="${path}" --dataset_name=${dset_name} --verbose=True --results_dir=${res_dir} --n_sup_vols=${perc} --split_number=${split}
```  
This will train the model and do a final test on the ACDC dataset. 
If you also want to test the results using the challenge server, after running the above command, you must run:   
```  
python -m test_on_acdc_test_set --RUN_ID="${run_id}"_${perc}_${split} --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${dpath} --experiment="${path}" --dataset_name=${dset_name} --verbose=False --n_sup_vols=${perc} --split_number=${split}
```  
and then submit the results as explained [here](https://acdc.creatis.insa-lyon.fr/description/databasesTesting.html).

Refer to the file `run.sh` for a complete example.


## Requirements
This code was implemented using TensorFlow 1.14.
We tested it on a TITAN Xp GPU, and on a GeForce GTX 1080, using CUDA 8.0, 9.0 and 10.2. 


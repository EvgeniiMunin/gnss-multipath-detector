cd /
cd opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/
#conda activate my_env
mo.py --data_type FP16 --framework tf --input_model /home/evgenii/Documents/01_These/gnss_signal_generator/saved_models/tf_model/tf_mp_model.pb --model_name ir_model --output_dir /home/evgenii/Documents/01_These/gnss_signal_generator/saved_models/ir_model --input_shape [1,40,40,2]


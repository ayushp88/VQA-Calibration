VQA Calibration
To run the multi-agent deliberation simulation:
python3 collaborative-calibration/deliberation_simulator.py \
-logging_level <logging_level> \
-input_dataset <input_dataset_name> \
-test_sample_size <test_size> \
-n_thread <number_of_threads> \
-group_size <group_size> \
--model_ensemble \
--agent_ensemble \
-logfile_path <logfile_path>

train_log_path=bart_base_$(date +"%d.%m.%Y_%H:%M:%S").log
nohup python ./train_bart_LM.py --is_debug=0 > ./training_logs/$train_log_path &
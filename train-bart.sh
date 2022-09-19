git_status=$(git status -s) 

if [ -n "$git_status" ]; then
    echo "You have uncommitted changes. Please commit them first."
    exit 1
fi

train_log_path=bart_base_$(date +"%d.%m.%Y_%H:%M:%S").log
nohup python train_bart_LM.py --is_debug > ./training_logs/$train_log_path &
echo "start training"
git_status=$(git status -s) 

if [ -n "$git_status" ]; then
    echo "You have uncommitted changes. Please commit them first."
    exit 1
fi

training_script=core.train_scripts.mpnet_training
training_logs=./training_logs/mpnet/
# clear dir
rm -rf $training_logs*

train_log_path=train_$(date +"%d.%m.%Y_%H:%M:%S").log
nohup python -m $training_script > $training_logs$train_log_path &

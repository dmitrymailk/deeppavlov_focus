deberta_training_script=core.train_scripts.debertav3_training
training_logs=./training_logs/debertav3/

debug() {
    echo "start debugging"
    python -m $deberta_training_script --debug_status 1
}

train() {
    echo "start training"
    git_status=$(git status -s) 

    if [ -n "$git_status" ]; then
        echo "You have uncommitted changes. Please commit them first."
        exit 1
    fi

    train_log_path=train_$(date +"%d.%m.%Y_%H:%M:%S").log
    nohup python -m $deberta_training_script > $training_logs$train_log_path &
}
small_train() {
    echo "start small training"

    train_log_path=train_$(date +"%d.%m.%Y_%H:%M:%S").log
    nohup python -m $deberta_training_script --debug_status 2 > $training_logs$train_log_path &
}

debug=0
persona=0
while getopts "d:p:" opt; do
	case $opt in
		d)
            debug=${OPTARG}
            ;;
        p)
            persona=${OPTARG}
            ;;
    esac
done

# clear dir
rm -rf $training_logs*

if [ $persona -eq 1 ]; then
    echo "start persona training"
    export WANDB_PROJECT=focus_persona_classification
fi

if [ $debug -eq 1 ]; then
    debug
elif [ $debug -eq 0 ]; then
    train
elif [ $debug -eq 2 ]; then
    small_train
fi

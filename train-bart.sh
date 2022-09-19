bart_training_script=core.train_scripts.bart_training

bart_debug() {
    echo "start debugging"
    python -m $bart_training_script --is_debug
}

bart_train() {
    echo "start training"
    git_status=$(git status -s) 

    if [ -n "$git_status" ]; then
        echo "You have uncommitted changes. Please commit them first."
        exit 1
    fi

    train_log_path=bart_base_$(date +"%d.%m.%Y_%H:%M:%S").log
    nohup python -m $bart_training_script > ./training_logs/bart/$train_log_path &
}

debug=0
while getopts "d:" opt; do
	case $opt in
		d)
            debug=${OPTARG}
    esac
done

if [ $debug -eq 1 ]; then
    bart_debug
elif [ $debug -eq 0 ]; then
    bart_train
fi
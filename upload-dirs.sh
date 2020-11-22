EXPERIMENT_NAME=$1
TIME="$(date -Iminutes)"
aws s3 sync results "s3://master-thesis-results/$EXPERIMENT_NAME-$TIME/results" --profile tomasruiz
aws s3 sync logs "s3://master-thesis-results/$EXPERIMENT_NAME-$TIME/logs" --profile tomasruiz

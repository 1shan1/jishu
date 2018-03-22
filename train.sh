export PYTHONPATH=./:/home/pengshanzhen/caffe/python:$PYTHONPATH
LOG=/media/pengshanzhen/bb42233c-19d1-4423-b161-e5256766be8e/new_people/model2/log-`date +%Y-%m-%d-%H-%M-%S`.log 
/home/pengshanzhen/caffe/build/tools/caffe train -solver solver.prototxt  2>&1  | tee $LOG $@







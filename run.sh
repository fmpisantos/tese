docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker build -t tfs_nima iqa/contrib/tf_serving
docker run -d --name tfs_nima -p 8500:8500 tfs_nima
source bin/activate

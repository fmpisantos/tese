docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker build -t tfs_nima iqa/contrib/tf_serving
docker run -d --name tfs_nima -p 8500:8500 tfs_nima
export GOOGLE_APPLICATION_CREDENTIALS="thesis-312720-c0663e5d4e21.json"
source bin/activate

How to get Submodules:
git submodule update --init

Start docker on wsl ?
sudo dockerd

How to run: 
0. To reset docker containers:
   ```
   Stop all running containers: docker stop $(docker ps -a -q)
   Delete all stopped containers: docker rm $(docker ps -a -q)
   ```
1. Build the NIMA TFS Docker image `docker build -t tfs_nima iqa/contrib/tf_serving`
2. Run a NIMA TFS container with `docker run -d --name tfs_nima -p 8500:8500 tfs_nima`
3. Install python dependencies
    ```
    virtualenv -p python3 .
    source bin/activate
    pip install -r requirements.txt
    ```
4. python3 main.py
    ```
    optional arguments:
    -h, --help              Show this help message and exit
    -fps FPS, --fps FPS     Frames per sec.
    -is IMGSEC, --imgSec    IMGSEC Seconds per image.
    -ts TSEC, --tSec TSEC   Seconds per transiction.
    -ni NIMAGES, --nImage   NIMAGES Number of images to show.
    ```

https://bitbucket.org/novaphotoapp/imagens_similares/src/master/

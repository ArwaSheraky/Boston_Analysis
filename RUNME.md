* To run the file normally: `python3 -W ignore boston_analysis.py`
* To run the file using docker:
    * `cd` to the repository directory.
    * Build a docker image with the existed Dockerfile:
        * `docker build -t arwasheraky/cebd1160_project ./`
    * Run the image using mount:
        * `docker run -ti -v $PWD:$PWD -w $PWD arwasheraky/cebd1160_project`
        * OR, if using Windows: `winpty docker run -ti -v /${PWD}://boston_analysis -w //boston_analysis arwasheraky/cebd1160_project`
    * Inside the image: `python3 -W ignore boston_analysis.py`

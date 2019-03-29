* To run the file normally: `python3 -W ignore boston_analysis.py`
* To run the file using docker:
    * `cd` to the repository directory.
    * Build a docker image with the existed Dockerfile:
        * `docker build -t arwasheraky/boston_analysis ./`
    * Run the image using mount:
        * `docker run -ti -v $PWD:$PWD -w $PWD arwasheraky/boston_analysis`
        * OR, if using Windows: `winpty docker run -ti -v /${PWD}://boston_analysis -w //boston_analysis arwasheraky/boston_analysis`
    * Inside the image: `python3 -W ignore boston_analysis.py`

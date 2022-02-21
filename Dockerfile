FROM python:3.6

ADD ./examples/docker-agent /agent
ADD ./pommerman /pommerman

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman_il
RUN cd /pommerman_il && pip install .
# end @TODO

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /pommerman
ENTRYPOINT ["python"]
CMD ["main.py"]

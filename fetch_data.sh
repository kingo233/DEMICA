#!/bin/bash

echo -e "\nDownloading insightface models..."
mkdir -p ~/.insightface/models/
if [ ! -d ~/.insightface/models/antelopev2 ]; then
  wget -O ~/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
  unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2
fi
if [ ! -d ~/.insightface/models/buffalo_l ]; then
  wget -O ~/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1"
  unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
fi
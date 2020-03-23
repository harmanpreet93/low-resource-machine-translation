#!/bin/bash
while :
do
  echo "Removing logs"
  rm -r ../log/*
  rsync -avzh guest140@helios3.calculquebec.ca:/home/guest140/team08/log .
  sleep 360
done

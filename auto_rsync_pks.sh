#!/bin/bash
while true; do

rsync --exclude build --exclude .git --exclude dist --exclude tmp -avc . pks:~/projects/RBMonGPU/
beep

inotifywait -r -e create -e delete -e modify -e moved_to -e moved_from --exclude .git .
sleep 1
 
done

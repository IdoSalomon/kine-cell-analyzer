ffmpeg -f image2 -r 5 -i Scene1Interval%%03d_PHASE.png -vcodec libx264 -b 1500k -vf normalize=blackpt=black:whitept=white:smoothing=0 -y movie.mp4
pause

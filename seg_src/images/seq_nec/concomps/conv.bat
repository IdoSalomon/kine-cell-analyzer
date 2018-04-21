ffmpeg -f image2 -r 5 -i Scene1Interval%%03d.png -vcodec libx264 -b 800k -y movie.mp4
pause

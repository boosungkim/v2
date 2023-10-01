---
layout: post
title: Docker with MERN and Vite
date: 2023-07-23 00:53:00-0400
description: My experience with dockerizing a full stack web app for the first time.
categories: devops
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---
Code: [https://]()

I used Docker in the past while working on ML research with professors from my college, but this was the first time I used it with the MERN tech stack. It was mostly similar, but I thought I would log different commands and debugging methods I used.

## General Code


## Client side containerization

docker build . -t boosungkim/deepfiction-client
docker run -p 5173:80 -d boosungkim/deepfiction-client

## Server side containerization



## Explain dockerfile
port 80 explained

## Removing docker images
docker stop CONTAINERID
docker rm CONTAINERID
docker rmi -f IMAGEID

## check
docker images
check images

docker ps
check running containers

going into docker

docker exec -it CONTAINERID /bin/bash
enter bash of docker

docker exec -it IMAGEID cat /etc/nginx/sites-enabled/default

docker port IMAGEID
check output port

docker exec -it CONTAINERID ls -l /var/www/html/
check contents of /var/www/html/

docker exec -it CONTAINERID ps aux | grep nginx

vim /usr/local/etc/nginx/nginx.conf
had to change the location of index.html (but change it so that i dont have to)

originally COPY --from=client /usr/src/client/dist /var/www/html
fixed by add /



sudo lsof -i :5173
sudo kill numbers

## Subheading
<img src="/assets/img/blogs/2023/2023-06-25-rcnn-explained-part1/classification-vs-detection.jpg"  width="500">
---
layout: post
title: The time I coded Chess with pen and paper in the Korean Army
date: 2023-06-17 15:53:00-0400
description: The story of how I coded Chess on pen and paper in the Korean Army.
categories: software story
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---

This is a story about how I brought back and used my Custom Chess Engine during my service in the Korea Army. With limited resources and immense determination, I overcame obstacles to bring the joy of chess to fellow soldiers. Who knows, perhaps my creation still thrives today?

<img src="/assets/img/blogs/2023/2023-06-17-chess-engine-comeback/army-time.jpeg"  width="600">

## My military service
As many of you probably know, the military service is mandatory for most men in Korea. I enlisted in October 2021 and finished my service as a Sergeant this April.

<img src="/assets/img/blogs/2023/2023-06-17-chess-engine-comeback/army-time.jpeg"  width="600">

As a mandatory duty, I enlisted in the Korean Army in October 2021 and completed my service as a Sergeant in April of this year. Assigned as a Data Analyst for the Tunnel Neutralization Team in the Demilitarized Zone (DMZ), I engaged in data processing and DMZ missions throughout my tenure.

Our base, situated outside the civilian zone, imposed numerous restrictions. Notably, we lacked modern facilities, including a PC room typically found in other bases.

## The spark of an idea
Playing chess on military work computers was strictly prohibited due to security concerns. However, there were old recreational computers available for soldiers' use during downtime—albeit without internet access.

In the evening, we were allowed 2-3 hours with our phones (on non-surveillance shifts). While soldiers utilized this time as they pleased, it was a fraction of the day, limiting options for entertainment. One day, upon learning of my background in Computer Science, a superior casually suggested creating games for these recreational computers.

That moment sparked my idea. I had previously developed a mostly finished Python-Chess project. Why not transfer the code to a military computer? Thus, I devoted the first half of 2022 to migrating my Chess project to a military machine.

## The mighty pen
Without internet access on the recreational computers, I couldn't simply download the files from GitHub. However, I still had access to my phone during the evenings.

Initially, I intended to bring my phone into the recreational room and copy the code directly onto the computers. However, I discovered that phones were not allowed there due to the proximity of a [REDACTED] area.

Can you guess what I did? With approximately 2000 lines of code in my Chess project, I spent an hour or two every day for three months painstakingly copying code from GitHub onto paper and then transcribing it onto the recreational computers. Over 2000 lines of code.

On days without evening work shifts, I would receive my phone, find a hilltop bench with a serene river view, and meticulously transcribe the code, line by line, word by word.

There were moments when I accidentally wrote down the wrong line, resulting in hours of debugging. There were also days filled with emergency situations, like the threats of nuclear experimentation from North Korea last year. Despite the challenges, I persisted with unwavering determination.

After months of resolute work, the transfer was complete, and the recreational army computer officially hosted the Chess game.

## Entertaining a whole base
<img src="/assets/img/blogs/2023/2023-06-17-chess-engine-comeback/pc-room-army.jpg"  width="600">

Transferring 2000 lines of code was no small feat, but witnessing soldiers play my game to unwind after arduous work made it all worthwhile.

While I relish the problem-solving aspect of software development, the joy is multiplied when others actively use and enjoy my creations. Based on my rough estimate during service, I believe dozens of people played my game at least once.

Since my honorable discharge this year, I haven't received much news about my former base. However, I like to imagine that future generations of soldiers will continue to relish the fruits of my labor.

## What I learned
1. Listen to what people want

While it's fulfilling to pursue projects that personally resonate, the true essence of product development lies in satisfying the desires of consumers. I'm grateful for recognizing the lack of daytime recreational activities based on my supervisor's passing remark. Despite his unsuspecting nature, I employed my coding expertise to identify and resolve a problem.

2. Resistance and Resilience

This "code transfer" was far from the most complex task in my short CS career. However, the hurdles I encountered along the way made it one of the most challenging endeavors.

Coding within a controlled school environment is relatively straightforward, with minimal resistance. Yet, coding in the army exposed me to real-world obstacles. From the absence of internet and time constraints to skeptical commanders questioning my notebook filled with pages of code, I felt like I was navigating one fiery hoop after another.

I would imagine that coding outside of school is generally like this. During these hurdles, it is important to stay resilient and keep moving forward, one step at a time.

3. Better code

After revisiting my project code months after its initial creation, I identified repeated code sections requiring refactoring. Thanks to this project, I developed the habit of code reviewing, which I believe will prove invaluable in the future.

4. Speeding things up

Just before departing the base, a fellow soldier thanked me for my project and suggested speeding up the game as the Chess AI took a bit too long to make decisions. Originally a Python Chess project, I viewed this as an opportunity to learn C++ and optimize my code for increased speed.

I hope you enjoyed my story of coding with pen and paper for months in the Korean army, bringing the joy of Chess to fellow soldiers.

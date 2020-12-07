---
title: "Baby Berry Reinforcement Learning Environment"
date: 2020-12-06
tags: [reinforcement learning, Q-learning, double-Q-learning, SARSA, OOP]
mathjax: true
classes: wide
excerpt: "Solving a custom made reinforcement learning problem using a variety of algorithms"
---

[//]: # "If you want to skip to the finished product follow [this link](https://filipino-food-classifier.onrender.com/)."
If you would prefer to learn about this project by reading the code [go here](https://github.com/sjhatfield/babyberry). In order to revisit concepts and learned earlier in the year taking Georgia Tech's fantastic graduate course in reinforcement learning, I decided to develop my own learning environment and solve it using a variety of RL algorithms.

## Introduction to the Environment

Taking insipiration from my young son who absolutely adores eating all kinds of berries, this environment is made up of a rectangular grid which the baby may move around attempting to collect and eat berries. The berries being spherically shaped, are able to roll around the environment making the game a little more challenging for the baby. Finally, to make the task even more challenging for the intrepid berry hunter there may be a parent present marching around the grid trying to stop the baby from consuming all the expensive berries.

Here is an example of the baby (in green) moving randomly around the environment. The berries (in purple) have been given a random movement probability of 50%. There is a parent (in blue) who is also moving randomly around the environment with probability 50%. Parents (or dads as I call them in the code) are referred to as dumb if they just move randomly.


![Baby moving randomly against dumb dad](https://github.com/sjhatfield/babyberry/blob/main/images/random-dumb.gif)

The environment is designed so that each individual berry may be given a different movement probability and starting position, or they may be assigned randomly.

The dad may be initialized to be *smart* which means they move towards the baby at each time step. Here is an example of a smart dad moving again with probability 50%.


![Baby moving randomly against smart dad](https://github.com/sjhatfield/babyberry/blob/main/images/random-smart.gif)
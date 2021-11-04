## Pendulum
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

Pendulum is classical control theory model. Pendulum start at random position. The goal of agent is to swing it up until it stays up.

### Observation vector
Is vector consisting of [cos(angle), sin(angle), speed].
### Action vector
Is vector [torq] to apply to pendulum. Is clipped by max torque value before applying.
### How reward is calculated
Reward is **costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)** where _u_ is clipped torque, _th_ is current angle, _thdot_ is current speed.
### How the initial conditions are varied by reset()
The reset generates state as vector of [_theta_, _thdot_] in range of (-pi, pi) for _theta_ and (-1, 1) for thdot.
### Termination condition
The termination conditions in code is only when number of steps exceeded 200. But, we may define additional constraint, like is when the pole will stand up straight, which means cost is around 0. For example say that if pole is **-0.05 <= cost <=0.05** for 60 steps, then we can terminate.



## MountainCar
https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

The car is one-dimensional track. As car engine is not strong enough to scale the mountain in one pass, the agent needs to move back and forth to accelerate the car.

### Observation vector
Is vector consisting of [car_position(x, y), car_velocity(dx, dy)]
### Action vector
Is a discrete value : 0 accelerate to left, 1 no acceleration, 2 accelerate to right.
### How reward is calculated
Reward 0 if the agent reached the flag, reward -1 in other steps
### How the initial conditions are varied by reset()
The position of car is reset to uniform value from [-0.6, -0.4], velocity is assigned to 0
### Termination condition
Is when car position is more than 0.5 or episode length is greater than 200.
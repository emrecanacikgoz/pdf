# Robot-Language Project

## Previous Works

### Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language [2204](https://arxiv.org/abs/2204.00598)

- They use LLMs to prompt other pretrained models for information that can build context about what is happening in a scene and make predictions about multimodal tasks.
- This approach can achieve state-of-the-art performance in zero-shot image captioning and video-to-text retrieval tasks.

### Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [2204](https://arxiv.org/abs/2204.01691)

- They ground the [PaLM](https://arxiv.org/abs/2204.02311) language model in a robotics affordance model to plan long horizon tasks.
- One needs to have both an LLM that can predict the sequence of steps to complete long horizon tasks and an **affordance model** representing the skills a robot can actually do in a given situation.

### Value Function Spaces: Skill-Centric State Abstractions for Long-Horizon Reasoning [2111](https://arxiv.org/abs/2111.03189)
- They showed that the value function in reinforcement learning (RL) models can be used to build the affordance model — an abstract representation of the actions a robot can perform under different states.
- This leads to connect long-horizons of real-world tasks, like “tidy the living room”, to the short-horizon skills needed to complete the task, like correctly picking, placing, and arranging items.

### Inner Monologue: Embodied Reasoning through Planning with Language Models [2207](https://arxiv.org/abs/2207.05608)
- Having both an LLM and an affordance model doesn’t mean that the robot will actually be able to complete the task successfully.
- In that work, they closed the loop on LLM-based task planning with other sources of information, like human feedback or scene understanding, to detect when the robot fails to complete the task correctly.
- They showed that LLMs can effectively replan if the current or previous plan steps failed, allowing the robot to recover from failures and complete complex tasks like "Put a coke in the top drawer".
- With PaLM-SayCan, the robot acts as the language model's "hands and eyes," while the language model supplies high-level semantic knowledge about the task.

### Talking to Robots in Real Time [2210](https://arxiv.org/abs/2210.06407)
- While natural language makes it easier for people to specify and modify robot tasks, one of the challenges is being able to react in real time to the full vocabulary people can use to describe tasks that a robot is capable of doing.
- They demonstrated a large-scale imitation learning framework for producing real-time, open-vocabulary, language-conditionable robots.
- They released [Language-Table](https://github.com/google-research/language-table), the largest available language-annotated robot dataset, which we hope will drive further research focused on real-time language-controllable robots.

### RT-1: Robotics Transformer for Real-World Control at Scale [2212](https://arxiv.org/abs/2212.06817)
- While we often take these physical skills for granted, executing them hundreds of times every day without even thinking, they present significant challenges to robots. For example, to pick up an object, the robot needs to perceive and understand the environment, reason about the spatial relation and contact dynamics between its gripper and the object, actuate the high degrees-of-freedom arm precisely, and exert the right amount of force to stably grasp the object without breaking it. 
- The difficulty of learning these low-level skills is known as Moravec's paradox: reasoning requires very little computation, but sensorimotor and perception skills require enormous computational resources.
- Inspired by the recent success of LLMs, which shows that the generalization and performance of large Transformer-based models scale with the amount of data, we are taking a data-driven approach, turning the problem of learning low-level physical skills into a scalable data problem.
- In that work, they trained a robot manipulation policy on a large-scale, real-world robotics dataset of 130k episodes that cover 700+ tasks using a fleet of 13 robots and showed the same trend for robotics — increasing the scale and diversity of data improves the model ability to generalize to new tasks, environments, and objects.
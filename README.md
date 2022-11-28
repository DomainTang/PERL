## Projected Space Exploration for Reinforcement Learning (PERL) 

1. Use on-line reinforcement learning to generate data
2. Sample a bitch of data from replay buffer
3. Solve the best projection matrix by maximizing the entropy of the post projected matrix
4. Select shared goals basing on the count-based method
5. Update replay buffer to train the total policy

## Note

The code is a version of PERL which is fit for discrete environment such as Push-Box,
Room,Secret-Room and so on. The version fitting for `StarCraftII` is published in another link. 

## Run

You should edit the configuration of the main function as follows. The name of test 
environment, the Learning rate of exploration policy and active policy
, the dictionary of log and the number of seed.
And you can also adjust other parameters in get_args().
For example, you can run PERL in `room30` with dense reward. 
The result will be stored in ./log/exp_batch_no_name/PERL_ROOM_0/.

    --env=room30_ckpt
    --alpha1=0.1
    --alpha2=0.05
    --exp_name= PERL_ROOM
    --seed= 0

## Acknowledge

1.[CMAE:Cooperative Exploration for Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/2107.11444)

  [The code Link](https://github.com/IouJenLiu/CMAE)

2.EITI_EDTI: [Influence-Based Multi-Agent Exploration](https://arxiv.org/abs/1910.05512)
  
  [The code Link](https://github.com/TonghanWang/EITI-EDTI)


## Statement

The code is not absolutely normative, which is only for learning and testing!  
Welcome to communicate and discuss!

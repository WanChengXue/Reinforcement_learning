'''
crossEntropy_method
初始的时候策略是随机的,即有一个state和action的table，然后所有值都是1/|action_space|
这个文件里面所谓的crossentropy_method就是先sample很多条trajectories， 并且返回对应的state_array, action_array, reward_array
对此次sample出来的一个集合，然后设置一个threshold，然后选出reward大于这个value的Trajectories，然后返回action和state
统计一下s,a出现的pair(需要正则化),将所有值除以行和，如果在这些Trajectories里面没有这个状态，那么对应的行全部给1/|action_space|
然后通过策略更新：
policy = (1-learning_rate)* policy + learning_rate * new_policy, 其实相当于加权，这个可以保证新计算出来的policy的行和一定为1
不停的循环下去，直至算法收敛
'''

‘’‘
crossEntropy_method + deep neural network version

明天任务 改成pytorch版本，这个scikit-learn模块不熟悉

在这个文件中，同样是sample，但是对于action的选择是通过一个神经网络来决定的
首先sample大量的Trajectories，然后对一些品相比较好的挑出来，就是reward相对比较大的样本，然后训练了一个网络，此处算是一个分类问题，
然后采用交叉熵进行修正神经网络的参数，X对应于box的vector，每一个样本是一个四维坐标，然后y是label，表示action，要么是0，要么是1，
对应于小车要么朝着左边走，要么朝着右边走，训练出来的神经网络优点在于其能够处理连续空间的值，就是说，需要每次sample的状态都是不一样的
但是可以拟合出一个比较好的网络，这样面对新的样本值，就能给出准确的判断。
'''
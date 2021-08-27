import tensorflow as tf

""" A.I, M.L, N.N.
    Artificail Intelligence: Any program that can autmate a tasks of human itellect, simple or complex. It can just
    be an algo that follows a set of simple rules like An algo that figures out the shortest path to somewhere.

    Machine Learning: giving an algorithm data and the expected result and letting it figure out the rules to get
    to the outcome by itself
    
    Neural Networks: multi layered machine learning where the network is given data and expected result. At
    each layer the networks transforms the data somehow to find new features that can be used as rules and basis
    its decision upon the differen features it has extracted
     """

""" FEATURES AND LABELS
    Features = input information data
    Labels = out/result accoriding to feature
    """

"""TYPES OF M.L
    Supervised Learning: you have feature and accoding labels pass both into newtork for training and it find rules
    if result is wrong just tweak the rules a little bit
    Unsupervised Learning: only have features and net comes up with the labels itself
    Reinforcement Learning: no data, your have Agent, Environment and Reward.Exp agent:player, enviroment:ground,
    goal:getting player to point x. Algo tries out different step in random directions, if it gets closer to flag it
    gets a reward, if farther away it gets negative reward. starts exploring to max reward.
    """

"""TF GRAPHS SESSION TENSOR
    Graphs: defines variables and how they are computed (dependencies), does NOT computes them only defines how
    Session: starts executing the graph
    Tensor: generalization of vectors/matrices to potentially higher dimensions. n-dimesional arrays of base datatypes
    each tensor represents a partialy defined computation that will eventually define a value
    
    tensorflow build a graph of tensor objects that detail how they are related and lets you run part/whole graph
    
    SINGLE ELEMTENT TENSOR
    newTensor = tf.Variable(324, tf.int16)//stores one value, called SCALAR

    MULTI RANK/DEGREE(NUMBER OF DIMENSIONS) TENSOR
    newTensor = tf.Variable([1,2,3,4,5], tf.int16)//one dimensional tensor
    newTensor = tf.Variable([[1,2,3,4,5][1,2,3,4,5]], tf.int16)//two dimensional tensor

    GET TENSOR RANK/SHAPE/TYPE
    tf.rank(myTensor)
    above returns follow: <tf.Tensor(rank, shape=(), dtype= >

    TENSOR SHAPE
    shape is the amount of elements that exist in each dimension
    myTensor.shape

    RESHAPE DATA: FLATEN(CHANGE TO LESS DIMENSIONS), DEEPEN(CHANGE TO MORE DIMENSION)
    newTensor = tf.reshape(oldTensor, [1,2,3])// second param represents the shape this exp = 1list with 2list with 3elements
    if last element in shape arr param is -1 [1, 2, -1] it figures out how many elements go in each list
     """
"""TYPES OF TENSORS
    TYPES: Variable, Constant, Placeholder, SparseTensor
    only Variable tensor is changable rest stays the same
    """
"""SESSION
    with tf.Session() as sess:
        myTensorName.eval() 

    this evaluates the tensor called myTensorName that was stored in the default graph based on its definition
    """

    
###################################################LEARNING ALGORITHMS####################################################################################################

"""LINEAR REGRESSION
    used to predict numeric values based on given a x value what is its y value
    |
    |                x
    |
    |             x
    |
    |      x
    |   x
    |________________________
    find the line of best fit. this doesnt only work with 1 input 1 output x,y but also works with 8 inputs giving one output
    
    """
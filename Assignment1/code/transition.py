class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1
        # END if

        # Get the next node of buffer
        b = conf.buffer[0]
        # Pop the next node on the stack
        s = conf.stack.pop(-1)
        # Add the arc (b, L, s)
        conf.arcs.append((b, relation, s))
#        print "{0} <- {1}".format(b, s)

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))
#        print "{0} -> {1}".format(idx_wi, idx_wj)

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.stack:
            return -1
        # END if

        # Pop the stack
        conf.stack.pop(-1)
#        print "pop {0}".format(s)
#        print "Stack Length: {0}".format(len(conf.stack))


    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1
        # END if

        # Pop buffer
        b = conf.buffer.pop(0)
        # Push onto stack
        conf.stack.append(b)
#        print "shift {0}".format(b)
#        print "Buffer Length: {0}".format(len(conf.buffer))

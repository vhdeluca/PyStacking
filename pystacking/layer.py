from bisect import insort
from .sublayer import SubLayer


class Layer(object):
    """ This class represents a single level of the stacking. """

    def __init__(self, level):
        """ The level defines the execution order of this layer. """
        self.level = level
        self.sublayers = []

    def __eq__(self, other):
        return self.level == other.level

    def __lt__(self, other):
        return self.level < other.level

    def insert_sublayer(self, sublevel=None):
        """ Insert a sublayer into this layer.
        All sublayers must have different execution orders.
        """

        if sublevel is not None:
            sublayer = SubLayer(self.level, sublevel)
        else:
            sublayer = SubLayer(self.level, self._get_lowest_sublevel())

        if sublayer not in self.sublayers:
            insort(self.sublayers, sublayer)
        else:
            raise ValueError("there is already a sublayer in this "
                             "layer with the same execution order!")

        return sublayer

    def _get_lowest_sublevel(self):
        """ Execution order starts from 0 (first to be executed). """
        if self.sublayers:
            return self.sublayers[-1].sublevel+1
        else:
            return 0

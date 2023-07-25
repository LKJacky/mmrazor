from mmcv.cnn.bricks.drop import DropPath

class MyDropPath(DropPath):

    def extra_repr(self) -> str:
        return str(self.drop_prob)

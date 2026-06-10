from processor.mind_processor import MINDProcessor
from processor.base_processor import Interactions


class MindTinyProcessor(MINDProcessor):
    TINY_TRAIN = 1000
    TINY_VALID = 100
    TINY_TEST = 100

    def load_interactions(self) -> Interactions:
        result = super().load_interactions()
        result.train_df = result.train_df.head(self.TINY_TRAIN).reset_index(drop=True)
        result.valid_df = result.valid_df.head(self.TINY_VALID).reset_index(drop=True)
        result.test_df = result.test_df.head(self.TINY_TEST).reset_index(drop=True)
        result[Interactions.train] = result.train_df
        result[Interactions.valid] = result.valid_df
        result[Interactions.test] = result.test_df
        return result

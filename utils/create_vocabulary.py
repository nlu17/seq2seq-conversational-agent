import numpy as np
import sys
from vocabulary import Vocabulary

VOC_SIZE = 20000
CORPUS_SIZE = -1


if __name__ == "__main__":
    voc = Vocabulary(voc_size=VOC_SIZE, corpus_size=CORPUS_SIZE)
    voc.init(sys.argv[1])
    voc.dump_to_file(sys.argv[2])

    voc2 = Vocabulary()
    voc2.load_from_file(sys.argv[2])
    print(voc2.parse("something nice is hapning to me"))

    print(voc2.sorted_voc[:10])
    print("====================")
    rank = np.where(voc2.voc["something"].one_hot == 1)[0][0]
    print(voc2.voc["something"], rank, voc2.sorted_voc[rank])

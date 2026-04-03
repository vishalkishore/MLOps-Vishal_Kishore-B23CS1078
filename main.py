from __future__ import annotations

import sys

from ques_1.train_q1 import main as q1_main
from ques_2.train_q2 import main as q2_main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "q2":
        sys.argv.pop(1)
        q2_main()
    else:
        q1_main()

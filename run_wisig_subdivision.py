from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "12")
os.environ.setdefault("OMP_NUM_THREADS", "12")

from functions.pipeline import run_post_training_open_set_steps
from functions.subdivision_pipeline import run_unknown_subdivision
from run_wisig import CHECKPOINT_PATH, build_config


# 是否先刷新边界挖掘、伪未知、OpenMax、融合阈值和开放集评估结果。
RUN_OSR_REFRESH = False

# 是否只复用已有闭集模型；细分入口通常不重新训练。
USE_EXISTING_CLOSED_SET_MODEL = True


def main() -> None:
    config = build_config()
    if RUN_OSR_REFRESH:
        run_post_training_open_set_steps(config, ckpt_path=CHECKPOINT_PATH if USE_EXISTING_CLOSED_SET_MODEL else None)
    run_unknown_subdivision(config, ckpt_path=CHECKPOINT_PATH if USE_EXISTING_CLOSED_SET_MODEL else None)


if __name__ == "__main__":
    main()

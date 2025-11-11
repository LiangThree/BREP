#!/bin/bash
# prefix_sample: 分析随机问题中三种模型前缀的回答效果
# prefix_reft_error: 分析reft错误问题中三种模型前缀的回答效果
# KL: 分析prefix回答过程中的KL散度
# bias: 分析不同训练结果中bias L2 Norm和结果的关系

condition=$1

if [ "$condition" == "prefix_sample" ]; then

    python Analyze/Prefix/prefix_answer.py
    python Analyze/Prefix/eval_sample_answer.py
    python Analyze/Prefix/draw_sample_fig.py

elif [ "$condition" == "prefix_reft_error" ]; then

    # python Analyze/Prefix/prefix_answer.py
    # python Analyze/Prefix/eval_reft_error_answer.py
    python Analyze/Prefix/draw_reft_error_fig.py

elif [ "$condition" == "bias" ]; then

    python Analyze/bias/bias_analyze.py

elif [ "$condition" == "KL" ]; then

    # python Analyze/KL/get_KL.py
    python Analyze/KL/draw_KL_fig.py

fi
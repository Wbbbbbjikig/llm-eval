"""规则模拟 LLM：无 API key 时用于打通全流程。
不同 Prompt 变体给予不同的规则完备度，模拟"更好的Prompt→更高准确率"。
真实提交时请用 src/evaluate.py 里的 call_real_llm 替换。"""
import json, re

HIGH_KW = ["失联","涉诉","冻结","呆账","造假","不符","多头","6家","10家","4家机构","杠杆","5倍","催收","连续6期","75%","1.1万"]
LOW_KW  = ["无逾期","从未逾期","零逾期","按时","全额还款","低","稳定","公积金","养老金","高净值","2套房","已结清"]
MID_KW  = ["白户","无任何信贷","历史","3年前","裁员","下滑","95%","最低还款","不固定","担保","容时","更换"]

def mock_llm(prompt: str, text: str, variant: str) -> str:
    """根据变体完备度返回判定。变体越完善，越少误判。"""
    h = sum(k in text for k in HIGH_KW)
    l = sum(k in text for k in LOW_KW)
    m = sum(k in text for k in MID_KW)

    # v1 朴素：只看关键词多数，易被"无逾期"误导（把高杠杆/多头判成低风险）
    if variant == "v1_zero_shot":
        if h >= 1 and "催收" in text: risk = "高风险"
        elif l > h: risk = "低风险"
        elif h > l: risk = "高风险"
        else: risk = "中风险"
        # 模拟格式不稳定：10%概率输出非JSON
        if len(text) % 10 == 0:
            return f"这个客户的风险等级是{risk}。"
    # v2/v3/v4 逐步改善
    elif variant in ("v2_role_rubric", "v3_few_shot", "v4_cot"):
        if h >= 1: risk = "高风险"
        elif m >= 1 and h == 0: risk = "中风险"
        elif l >= 1: risk = "低风险"
        else: risk = "中风险"
    # v5 组合：最完善，优先级 高>中特征>低
    else:
        if h >= 1: risk = "高风险"
        elif m >= 1: risk = "中风险"
        elif l >= 1: risk = "低风险"
        else: risk = "中风险"

    return json.dumps({"risk": risk, "reason": "基于规则判定"}, ensure_ascii=False)

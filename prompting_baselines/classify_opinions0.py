# _*_ coding:utf-8 _*_
import os
import openai
import pandas as pd
from dotenv import load_dotenv
import glob
import time
import argparse
from typing import List, Dict, Deque
from collections import deque
from sklearn.metrics import accuracy_score, f1_score, classification_report
# --- 1. 初始化设置 ---

# 加载环境变量 (OPENAI_API_KEY 和 OPENAI_BASE_URL)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_BASE_URL"):
    openai.api_base = os.getenv("OPENAI_BASE_URL")

# 配置您想使用的大模型
LLM_MODEL = "deepseek-v3"  # 或者 "gpt-4-turbo"
# 存放目标文件的文件夹
TARGET_DIR = "target1"
# 上下文窗口大小（即模型能看到的"前几句发言")
CONTEXT_WINDOW_SIZE = 5
# API调用之间的最小间隔（秒），以避免速率限制
RATE_LIMIT_DELAY = 0.5

# 定义六个分类标签
VALID_LABELS = {
    "Irrelevant", "New", "Strengthened",
    "Weakened", "Adopted", "Refuted"
}
# 为兼容旧代码/报告输出，提供一个有序的标签列表
VALID_LABELS_LIST = sorted(list(VALID_LABELS))


# --- 2. 大模型API调用函数 ---

def call_llm_api(messages: List[Dict[str, str]], model: str = LLM_MODEL) -> str:
    """
    调用大模型API并处理基本错误。
    (此函数改编自您提供的 tracker-act-9-30.py)
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0,  # 分类任务使用低温度以获得确定性结果
            max_tokens=50  # 标签很短，不需要很长的回复
        )
        label = response.choices[0].message['content'].strip()

        # 确保API的回复是我们想要的标签之一
        if label in VALID_LABELS:
            return label
        else:
            print(f"    [Warning] LLM返回了无效标签: '{label}'. 默认记为 'Irrelevant'.")
            return "Irrelevant"

    except Exception as e:
        print(f"    [Error] 调用 LLM API 时出错: {e}. 等待5秒后重试...")
        time.sleep(5)
        # 尝试重试一次
        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, temperature=0.0, max_tokens=50
            )
            label = response.choices[0].message['content'].strip()
            if label in VALID_LABELS:
                return label
            else:
                return "Irrelevant"
        except Exception as e2:
            print(f"    [Error] 重试失败: {e2}. 将此行标记为 'Error:API_Failure'.")
            return "Error:API_Failure"




# --- 3. 构建分类 Prompt ---

def get_classification_prompt(
        context_str: str,
        current_speaker: str,
        current_sentence: str,
        consistency: str
) -> List[Dict[str, str]]:
    """
    根据我们发现的规律，构建用于分类的 Prompt。
    [已优化：添加 Few-Shot 示例]
    """

    system_prompt = f"""
    你是一个课堂对话分析专家。你的任务是根据“上下文”（前面的发言）和“当前发言”之间的关系，对“当前发言”进行分类。
    你必须从以下六个标签中选择一个：

    1.  **Irrelevant (无关)**: 发言与讨论的主题无关，或者是关于对话状态的元评论（如 "我听不懂了"），或者是程序性发言（如老师管理课堂秩序）。
    2.  **New (新观点)**: 引入了一个在上下文中未曾出现过的新主张、新论点或新证据。
    3.  **Strengthened (强化)**: 为上下文中已有的某个观点提供支持、同意、证据或更详细的阐述。
    4.  **Weakened (削弱)**: 针对上下文中的观点提出疑问、反例、不同意见或指出其局限性，但没有完全否定它。
    5.  **Adopted (采纳)**: 发言人明确表示同意或接受了上下文中 *另一位* 发言人的观点（通常在 'switch' 状态下发生）。
    6.  **Refuted (驳斥)**: 发言人明确、直接地否定了上下文中的观点（通常在 'switch' 状态下发生，例如以 "No..." 或 "Yeah, but..." 开头）。

    ---
    **关键示例 (Examples):**

    示例 1: 学生采纳了老师的纠正
    Context:
    (New) Toby: An object that you can get stuff inside of it, I guess.
    (Refuted) T: Three dimensional is not flat.
    Current Utterance:
    - Speaker: Toby
    - Consistency: switch
    - Sentence: Not flat.
    分类:
    Adopted

    示例 2: 老师强化(复述)学生的观点
    Context:
    (New) Guy: It's the area around like the outside of it.
    Current Utterance:
    - Speaker: T
    - Consistency: switch
    - Sentence: The area of each of the surfaces, surface area.
    分类:
    Strengthened

    示例 3: 学生之间的直接反驳
    Context:
    (Strengthened) Erik: Yeah, it would.
    Current Utterance:
    - Speaker: Alan
    - Consistency: switch
    - Sentence: No, it doesn’t.
    分类:
    Refuted
    ---

    你的回复必须 *仅仅* 包含这六个标签中的一个词，不要有任何其他解释。
    """

    user_prompt = f"""
    **上下文 (Context):**
    {context_str if context_str else "（没有上下文，这是第一句发言）"}

    **待分类的发言 (Current Utterance):**
    - **发言人 (Speaker)**: {current_speaker} (注意：'T' 代表老师)
    - **发言人一致性 (Consistency)**: {consistency}
    - **句子 (Sentence)**: {current_sentence}

    请分类：
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()}
    ]


# --- 4. 主处理流程 ---

def process_spreadsheet(filepath: str):
    """
    处理单个 Excel 文件。
    """
    print(f"\n--- 正在处理文件: {filepath} ---")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"    [Error] 无法读取文件: {e}")
        return

    # 确保必需的列存在
    if 'Sentence' not in df.columns or 'Speaker' not in df.columns:
        print(f"    [Error] 文件必须包含 'Sentence' 和 'Speaker' 列。跳过此文件。")
        return

    # 使用 deque (双端队列) 高效地维护上下文窗口
    context_history: Deque[tuple[str, str, str]] = deque(maxlen=CONTEXT_WINDOW_SIZE)
    previous_speaker = None
    classification_results = []

    total_rows = len(df)
    for index, row in df.iterrows():
        current_sentence = str(row['Sentence'])
        current_speaker = str(row['Speaker'])

        print(f"     > 正在处理第 {index + 1} / {total_rows} 行: {current_sentence[:60]}...")

        # 1. 确定发言人一致性 (不变)
        if (previous_speaker is None) or (current_speaker == 'T' and previous_speaker == 'T'):
            consistency = "same"
        elif current_speaker == previous_speaker:
            consistency = "same"
        else:
            consistency = "switch"

        # 2. 格式化上下文 [已优化]
        # 将历史记录格式化为 (标签) 发言人: 句子
        context_lines = []
        for label, speaker, sentence in context_history:
            context_lines.append(f"({label}) {speaker}: {sentence}")
        context_str = "\n".join(context_lines)

        # 3. 构建 Prompt (不变)
        messages = get_classification_prompt(
            context_str, current_speaker, current_sentence, consistency
        )

        # 4. 调用 LLM 进行分类 (不变)
        label = call_llm_api(messages)
        classification_results.append(label)

        # 5. 更新上下文历史 [已优化]
        # 将 *刚刚生成的标签* 和当前句子一起存入历史
        context_history.append((label, current_speaker, current_sentence))
        previous_speaker = current_speaker

        # 6. 速率限制 (不变)
        time.sleep(RATE_LIMIT_DELAY)

    # 7. 保存结果
    df['opinion_classification'] = classification_results
    output_filename = filepath.replace(".xlsx", "_classified_v1.xlsx")

    try:
        df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"--- 处理完成！结果已保存到: {output_filename} ---")
    except Exception as e:
        print(f"    [Error] 保存文件失败: {e}")


def evaluate_classification(output_filepath: str):
    """
    读取已分类的文件，并将其与 'human_classification' 列进行比较。
    """
    print(f"\n--- 正在评测文件: {output_filepath} ---")
    try:
        df = pd.read_excel(output_filepath)
    except Exception as e:
        print(f"    [Error] 无法读取评测文件: {e}")
        return

    # 检查评测所需的列是否存在
    if 'human_classification' not in df.columns:
        print(f"    [Warning] 文件中缺少 'human_classification' 列，无法进行评测。")
        return
    if 'opinion_classification' not in df.columns:
        print(f"    [Error] 文件中缺少 'opinion_classification' 列。")
        return

    # 提取真实标签和预测标签
    y_true = df['human_classification'].astype(str)
    y_pred = df['opinion_classification'].astype(str)

    # 确保所有标签都在我们的 VALID_LABELS_LIST 中，这有助于 classification_report
    # (这只是为了让报告更整洁)
    all_labels = sorted(list(set(y_true) | set(y_pred) | set(VALID_LABELS_LIST)))

    print("\n--- 评测报告 (包含所有类别) ---")
    print(classification_report(y_true, y_pred, labels=all_labels, digits=3, zero_division=0))

    # 计算加权 F1-Score (处理标签不平衡问题)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"总体 Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"总体 Weighted F1-Score: {weighted_f1:.3f}")

    # --- 过滤掉 'Irrelevant' 后的评测 ---
    # 这通常是一个更有意义的指标，因为它专注于“观点演化”本身
    
    # 创建一个不包含 'Irrelevant' 的新 DataFrame
    df_filtered = df[
        (df['human_classification'] != 'Irrelevant') & 
        (df['opinion_classification'] != 'Error:API_Failure')
    ]
    
    if len(df_filtered) == 0:
        print("\n--- (没有非 Irrelevant 的行可供评测) ---")
        return

    y_true_filtered = df_filtered['human_classification'].astype(str)
    y_pred_filtered = df_filtered['opinion_classification'].astype(str)
    
    # 过滤后的标签
    filtered_labels = [label for label in all_labels if label != 'Irrelevant']

    print("\n--- 评测报告 (已过滤 'Irrelevant') ---")
    print(classification_report(y_true_filtered, y_pred_filtered, labels=filtered_labels, digits=3, zero_division=0))
    
    weighted_f1_filtered = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    print(f"过滤后 Accuracy: {accuracy_score(y_true_filtered, y_pred_filtered):.3f}")
    print(f"过滤后 Weighted F1-Score: {weighted_f1_filtered:.3f}")


def process_and_evaluate(filepath: str, force: bool = False):
    """
    对单个文件执行分类（调用 LLM）并随后评测（对比 human_classification）。
    如果已经存在输出文件且 force=False，则跳过分类并直接评测现有结果。
    """
    output_filename = filepath.replace('.xlsx', '_classified_v1.xlsx')

    if os.path.exists(output_filename) and not force:
        print(f"发现已存在的输出文件 {output_filename}（使用已有文件进行评测，若要强制重新分类请使用 --force）。")
    else:
        process_spreadsheet(filepath)

    # 调用评测（evaluate_classification 会检查 human_classification 列）
    if os.path.exists(output_filename):
        evaluate_classification(output_filename)
    else:
        print(f"未找到输出文件 {output_filename}，无法进行评测。请先运行分类或使用 --force 重新生成输出。")


# --- 5. 脚本执行入口 ---

def main():
    parser = argparse.ArgumentParser(description='Process and/or evaluate opinion classification files.')
    parser.add_argument('--mode', choices=['process', 'eval', 'process_eval'], default='process',
                        help='运行模式：process=仅分类，eval=仅评测（传入已分类文件），process_eval=先分类再评测')
    parser.add_argument('--file', '-f', help='单个目标 .xlsx 文件路径（若不指定则处理 TARGET_DIR 下所有文件）')
    parser.add_argument('--force', action='store_true', help='对已存在的输出文件强制重新分类（仅在 process 或 process_eval 有效）')

    args = parser.parse_args()

    # 如果需要调用 LLM（process 或 process_eval），则必须有 API key
    if args.mode in ('process', 'process_eval') and not openai.api_key:
        print("错误：OPENAI_API_KEY 未设置。请检查您的 .env 文件。")
        return

    if args.file:
        files = [args.file]
    else:
        excel_files = glob.glob(os.path.join(TARGET_DIR, "*.xlsx"))
        # 排除那些已经分类过（按旧逻辑），如果需要可调整为 '_classified_v1.xlsx'
        files = [f for f in excel_files if not f.endswith("_classified.xlsx")]

    if not files:
        print(f"在 '{TARGET_DIR}' 文件夹中未找到需要处理的 .xlsx 文件。")
        return

    print(f"准备在模式 '{args.mode}' 下处理 {len(files)} 个文件")

    for filepath in files:
        print(f"\n--- 处理: {filepath} ---")
        if args.mode == 'process':
            process_spreadsheet(filepath)
        elif args.mode == 'eval':
            evaluate_classification(filepath)
        elif args.mode == 'process_eval':
            process_and_evaluate(filepath, force=args.force)

    print("\n所有任务完成。")


if __name__ == "__main__":
    main()
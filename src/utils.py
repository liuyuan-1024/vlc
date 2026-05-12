# 十进制转二进制的基础函数
def decimal_to_binary(decimal_num, bit_length=3):
    """将十进制转换为固定长度的二进制字符串"""
    if decimal_num < 0:
        raise ValueError("仅支持非负整数")
    return format(decimal_num, f"0{bit_length}b")


# 动态映射：PAM符号 -> 格雷码比特串
def pam_to_gray_bits(symbol):
    """将 PAM-8 符号 (-7 到 7) 转换为对应的 3bit 格雷码字符串"""
    # 将符号电平映射为 0~7 的十进制索引 (-7->0, -5->1 ... 7->7)
    index = int((symbol + 7) // 2)

    # 将自然二进制索引转换为格雷码十进制值 (公式: Gray = n XOR (n右移1位))
    gray_dec = index ^ (index >> 1)

    # 使用十进制转二进制函数输出最终比特
    return decimal_to_binary(gray_dec, 3)


# 计算比特误码率（BER）
def calculate_ber(y_true_symbols, y_pred_symbols):
    bit_errors = 0
    # PAM-8 每个符号 3 个比特
    total_bits = len(y_true_symbols) * 3

    for true_sym, pred_sym in zip(y_true_symbols, y_pred_symbols):
        # 动态生成格雷码比特，不再依赖写死的字典
        true_bits = pam_to_gray_bits(true_sym)
        pred_bits = pam_to_gray_bits(pred_sym)

        # 逐位对比，不同则错误数 + 1
        bit_errors += sum(1 for a, b in zip(true_bits, pred_bits) if a != b)

    return bit_errors / total_bits

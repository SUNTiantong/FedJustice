def compute_probdpd(y_pred, s, target_s):
    # 筛选条件 s = target_s, Y = target_y
    indices = [i for i in range(len(y_pred)) if s[i] == target_s]
    if not indices:
        return 0.0
    # 预测为 1 的数量 / 满足条件的总数量
    pred_1_count = sum(y_pred[i] == 1 for i in indices)
    return pred_1_count / len(indices)
    #s=sensitive_features    
def DPD(s,y_pred):
    M_dpd = 0
    # 替换 'male' 为 1，'female' 为 0
    # s = [1 if gender == "male" else 0 for gender in s]
    prob_s0 = compute_probdpd(y_pred, s, target_s="Female")
    prob_s1 = compute_probdpd(y_pred, s, target_s="Male")
    M_dpd += abs(prob_s0 - prob_s1)
    return M_dpd
s = ["Male", "Female", "Male", "Female", "Male"]  # 性别敏感特征
y_pred = [0, 1, 1, 1, 0]  # 预测标签

# 计算 DPD
dpd_value = DPD(s, y_pred)
print("DPD:", dpd_value)
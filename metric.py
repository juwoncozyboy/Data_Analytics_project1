# metric.py
# 원하는 metric 함수를 추가하면 됨. 단, mae는 반드시 남겨둘 것

import torch

def rmse(output, target):
    with torch.no_grad():
        return torch.sqrt(torch.mean((output - target) ** 2)).item()
    

def mae(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item()


def mape(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs((output - target) / target)).item()

##추가한 매트릭스들##

#Hit Rate: 방향성 정확도로, 모델이 환율이 상승할지 하락할지의 방향을 얼마나 정확하게 예측했는지를 측정합니다.
def hit_rate(output, target):
    hits = torch.sign(output[1:] - output[:-1]) == torch.sign(target[1:] - target[:-1])
    return torch.mean(hits.float())

#R-squared (결정 계수):R-squared는 데이터의 분산 중 모델이 얼마나 잘 설명하는지를 나타내는 지표입니다.1에 가까울수록 모델이 데이터를 잘 설명하고 있다고 할 수 있습니다.
def r2_score(output, target):
    ss_res = torch.sum((output - target) ** 2)
    ss_tot = torch.sum((output - torch.mean(output)) ** 2)
    return 1 - ss_res / ss_tot

#Theil's U: Theil's U 통계는 실제값과 예측값의 비율을 기반으로 한 메트릭으로, 값이 0에 가까울수록 예측의 정확도가 높다는 것을 의미합니다
def theils_u(output, target):
    num = torch.sqrt(torch.mean((output - target) ** 2))
    denom = torch.sqrt(torch.mean(output ** 2)) + torch.sqrt(torch.mean(target ** 2))
    return num / denom
    

if __name__ == '__main__':
    import torch
    
    output = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    target = torch.tensor([2, 2, 3, 4, 7], dtype=torch.float32)
    
    print(f'RMSE : {rmse(output, target):.4f}')
    print(f'MAE  : {mae(output, target):.4f}')
    print(f'MAPE : {mape(output, target) * 100:.2f} %')
    print(f'R2   : {r2_score(output, target):.4f}')
    print(f'Theil\'s U : {theils_u(output, target):.4f}')
    print(f'Hit Rate : {hit_rate(output, target):.4f}')